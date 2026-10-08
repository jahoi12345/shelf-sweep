"""Shared stdlib process bridge for optional stage runtimes (integration ownership).

Model implementations stay in their workers' modules. A process initializes its
adapter once, serializes requests, and retains no Gemini credentials.
"""
from __future__ import annotations
import argparse
import asyncio
import atexit
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]

class ReaderProcess:
    def __init__(self, python, model_dir, package_dir, work_dir):
        self.python=Path(python).absolute();self.model_dir=Path(model_dir).resolve()
        self.package_dir=Path(package_dir).resolve();self.work_dir=Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True,exist_ok=True);self.lock=threading.Lock();self.process=None
        self.stderr=(self.work_dir/'stderr.log').open('ab')
        env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ['KEY','TOKEN','SECRET','CREDENTIAL','PASSWORD','AUTH'])}
        env.update(PYTHONPATH=str(self.package_dir)+os.pathsep+str(ROOT),MPLCONFIGDIR=str(self.work_dir/'mpl'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        try:
            self.process=subprocess.Popen([str(self.python),'-u','-m','providers.runtime','--serve-reader','--model-dir',str(self.model_dir),'--work-dir',str(self.work_dir)],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,bufsize=1,cwd=ROOT,env=env)
            self.setup=self._receive(60)
            if not self.setup.get('ready'):raise RuntimeError('Reader runtime did not initialize')
        except BaseException:self.close();raise
        atexit.register(self.close)

    def _receive(self,timeout):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout,selectors.EVENT_READ)
            if not selector.select(timeout):raise TimeoutError('Optional reader runtime timed out')
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('Optional reader worker exited; see stderr.log')
        response=json.loads(line)
        if response.get('error'):raise RuntimeError('Optional reader failure: '+response['error'])
        return response

    def _request(self,kwargs):
        with self.lock:
            if self.process is None or self.process.poll() is not None:raise RuntimeError('Reader runtime closed')
            message={k:v for k,v in kwargs.items() if k in ('image_paths','image_path','task')}
            for key in ('image_paths','image_path'):
                if key in message:message[key]=[str(x) for x in message[key]] if key=='image_paths' else str(message[key])
            cfg=kwargs.get('config') or {}
            message['config']={k:cfg.get(k) for k in ('ocr_evidence_dir','source_turn_id','session_revision')}
            try:
                self.process.stdin.write(json.dumps(message)+'\n');self.process.stdin.flush()
                response=self._receive(45)
                response.setdefault('metrics',{}).update(transport='persistent_optional_runtime',worker_pid=self.process.pid,
                    model_setup_s=self.setup['setup_s'])
                return response
            except BaseException:self.close();raise

    async def __call__(self,**kwargs):
        if kwargs.get('task','read')!='read':raise ValueError('Reader process cannot accept detection requests')
        return await asyncio.to_thread(self._request,kwargs)

    def close(self):
        process,self.process=self.process,None
        if process:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=3)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=3)
            for stream in (process.stdin,process.stdout):
                if stream:stream.close()
        if not self.stderr.closed:self.stderr.close()


def serve_reader(args):
    # The literal reader is used directly; no semantic reading or model code here.
    from providers.easyocr_reader import EasyOCRAdapter
    adapter=EasyOCRAdapter(args.model_dir,evidence_dir=Path(args.work_dir)/'responses')
    print(json.dumps({'ready':True,'setup_s':adapter.setup_s,'runtime':adapter.runtime}),flush=True)
    for line in sys.stdin:
        try:
            request=json.loads(line);evidence_dir=(request.get('config') or {}).get('ocr_evidence_dir')
            if evidence_dir:
                adapter.evidence_dir=Path(evidence_dir);adapter.evidence_dir.mkdir(parents=True,exist_ok=True)
            response=asyncio.run(adapter(**request))
        except Exception as exc:response={'error':type(exc).__name__+': '+str(exc)[:200]}
        print(json.dumps(response,ensure_ascii=False),flush=True)
    adapter.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--serve-reader',action='store_true');parser.add_argument('--model-dir',required=True);parser.add_argument('--work-dir',required=True)
    args=parser.parse_args();serve_reader(args)
