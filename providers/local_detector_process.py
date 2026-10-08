"""Persistent detection-only local worker across Python runtimes, no OCR/voice."""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
_LEGACY_ROOT=REPO/'runs/phase2_reader_comparison_v1'   # older layout, used only when .runtime has no detector
DEFAULT_ROOT=_LEGACY_ROOT if (_LEGACY_ROOT/'venv/bin/python').is_file() and not (REPO/'.runtime/venv/bin/python').is_file() else REPO/'.runtime'


# Default: extended non-book prompts, every frame. 'baseline' stays selectable.
DEFAULT_PROFILE = 'fast_items'


class LocalDetectorProcess:
    """Initialize models once in optional runtime; serialize framed JSONL requests.

    Cancellation discards caller publication but drains the in-flight response so
    a later frame cannot receive it. The session layer owns revision validation.
    Close this adapter on app shutdown. It returns detector objects, never OCR.
    """
    def __init__(self,*,python=None,weights_dir=None,work_dir=None,timeout_s=60,profile=None):
        from providers.local_detector import PROFILES
        profile = profile or os.getenv('LIBRARY_LOCAL_DETECTOR_PROFILE', DEFAULT_PROFILE)
        if profile not in PROFILES:raise ValueError('Unknown local detector profile')
        # Keep the venv invocation path: resolving its symlink runs the base
        # interpreter and silently loses the venv's optional dependencies.
        self.python=Path(python or DEFAULT_ROOT/'venv/bin/python').absolute()
        self.weights_dir=Path(weights_dir or DEFAULT_ROOT/'models').resolve()
        self.work_dir=Path(work_dir or DEFAULT_ROOT/'process_config').resolve()
        self.timeout_s=timeout_s;self.profile=profile
        self._lock=asyncio.Lock();self._process=None;self._stderr_task=None;self._tasks=set();self._closed=False
        self.stderr_tail=b'';self.setup_metrics=None
        self.completed_usage=deque(maxlen=1024)

    async def _stderr(self,stream):
        while chunk:=await stream.read(4096):
            self.stderr_tail=(self.stderr_tail+chunk)[-65536:]

    async def _line(self):
        line=await asyncio.wait_for(self._process.stdout.readline(),self.timeout_s)
        if not line:raise RuntimeError('Local detector worker exited without a response')
        result=json.loads(line)
        if result.get('error_type'):raise RuntimeError('Local detector worker failure: '+result['error_type'])
        return result

    async def _start_locked(self):
        if self._closed:raise RuntimeError('Local detector adapter closed')
        if self._process is not None:return
        if not self.python.is_file():raise FileNotFoundError('Optional local detector Python missing')
        for filename in ['book-spine-best.pt','yolov8m-worldv2.pt','clip/ViT-B-32.pt']:
            if not (self.weights_dir/filename).is_file():raise FileNotFoundError('Required local model missing: '+filename)
        env={k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ['KEY','TOKEN','SECRET','CREDENTIAL','PASSWORD','AUTH'])}
        env.update(YOLO_CONFIG_DIR=str(self.work_dir/'yolo'),MPLCONFIGDIR=str(self.work_dir/'mpl'),
                   YOLO_AUTOINSTALL='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        (self.work_dir/'yolo/Ultralytics').mkdir(parents=True,exist_ok=True);(self.work_dir/'mpl').mkdir(parents=True,exist_ok=True)
        self._process=await asyncio.create_subprocess_exec(str(self.python),'-u','-m','providers.local_detector',
            '--serve','--weights-dir',str(self.weights_dir),'--profile',self.profile,
            cwd=REPO,env=env,stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,limit=4*1024*1024)
        self._stderr_task=asyncio.create_task(self._stderr(self._process.stderr))
        self.setup_metrics=await self._line()
        if self.setup_metrics.get('status')!='ready':raise RuntimeError('Local worker initialization did not complete')

    async def start(self):
        async with self._lock:
            try:await self._start_locked()
            except BaseException:
                await self._stop_process();raise
        return self.setup_metrics

    async def _stop_process(self):
        process,self._process=self._process,None
        if process is not None:
            if process.returncode is None:
                process.terminate()
            try:await asyncio.wait_for(process.wait(),3)
            except asyncio.TimeoutError:
                process.kill();await process.wait()
        if self._stderr_task:
            await self._stderr_task;self._stderr_task=None

    async def _request(self,image_path,request_id):
        queued=time.perf_counter()
        async with self._lock:
            try:
                await self._start_locked()
                self._process.stdin.write((json.dumps({'image_path':str(Path(image_path).resolve()),'request_id':request_id})+'\n').encode())
                await self._process.stdin.drain()
                result=await self._line()
                result.setdefault('metrics',{}).update(adapter_wall_s=time.perf_counter()-queued,
                    worker_pid=self._process.pid,transport='persistent_local_JSONL',model_setup_s=self.setup_metrics['setup_s'])
                self.completed_usage.append(dict(result['metrics']))
                return result
            except BaseException:
                await self._stop_process();raise

    async def __call__(self,*,image_path,request_id=None,**kwargs):
        if kwargs.get('stage') not in (None,'detect'):
            raise ValueError('Local detector can only serve the detect stage, never OCR')
        task=asyncio.create_task(self._request(image_path,request_id));self._tasks.add(task)
        def done(t):
            self._tasks.discard(t)
            if not t.cancelled():t.exception()  # Drain errors after a caller cancels.
        task.add_done_callback(done)
        return await asyncio.shield(task)

    async def aclose(self):
        self._closed=True
        for task in list(self._tasks):task.cancel()
        await asyncio.gather(*self._tasks,return_exceptions=True)
        async with self._lock:await self._stop_process()

    def drain_usage(self):
        """The session layer may persist started work even when its caller was cancelled."""
        result=list(self.completed_usage);self.completed_usage.clear();return result


async def smoke(args):
    from providers.local_detector import prepare_run
    from pipeline.detect import _detect,save_json,read_json
    root=Path(args.out).resolve();manifest=prepare_run(Path(args.manifest).resolve().parent,root)
    adapter=LocalDetectorProcess(python=args.python,weights_dir=args.weights_dir,work_dir=root/'process_config',profile=args.profile)
    try:
        setup=await adapter.start()
        result=await _detect(manifest,root,{'image_request':adapter,'frame_ids':[args.frame_id]})
        save_json(root/'adapter_smoke.json',{'setup':setup,'metrics':read_json(result)['metrics'],
            'actual_API_requests':0,'profile':args.profile,'runtime':str(adapter.python),'weights_dir':str(adapter.weights_dir)})
        return result
    except Exception as exc:
        save_json(root/'adapter_smoke.json',{'status':'unavailable','failure_type':type(exc).__name__,
            'actual_API_requests':0,'runtime':str(adapter.python),'stderr_tail':adapter.stderr_tail.decode(errors='replace')})
        raise
    finally:await adapter.aclose()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    p.add_argument('--frame-id',default='f00066');p.add_argument('--python');p.add_argument('--weights-dir')
    from providers.local_detector import PROFILES
    p.add_argument('--profile',choices=PROFILES)
    args=p.parse_args();print(asyncio.run(smoke(args)))

if __name__=='__main__':main()
