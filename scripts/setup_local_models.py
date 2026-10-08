"""Pinned local setup with checked resumable downloads and per-step timing.

No API calls or credential output. Minimal mode deliberately disables detection;
full mode adds YOLO/CLIP + EasyOCR/CRAFT. Florence remains separately optional.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CLIP_COMMIT = 'a1d071733d7111c9c014f024669f959182114e33'
RAPID_BASE = 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/'
RAPID_MODELS = [
 ('PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx','4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae'),
 ('PP-OCRv5/rec/latin_PP-OCRv5_rec_mobile.onnx','b20bd37c168a570f583afbc8cd7925603890efbcdc000a59e22c269d160b5f5a'),
 ('PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx','e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c')]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def fetch(url, destination, expected, *, timeout=900, opener=urllib.request.urlopen):
    """Keep partial bytes on interruption; restart if server ignores Range.

    Only atomically publish a destination after the recorded SHA256 matches.
    A 416 can mean the retained part is already complete: verify before retry.
    """
    dest=Path(destination);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.is_file() and digest(dest)==expected:
        print(f'cached: {dest.name}',flush=True);return
    part=dest.with_name(dest.name+'.part');start=time.monotonic();last_error=None
    for attempt in range(3):
        if part.is_file() and digest(part)==expected:
            part.replace(dest);return
        offset=part.stat().st_size if part.exists() else 0
        req=urllib.request.Request(url,headers={'User-Agent':'library-setup/1','Range':f'bytes={offset}-'} if offset else {'User-Agent':'library-setup/1'})
        try:
            with opener(req,timeout=min(30,timeout)) as response:
                resumed=offset and response.status==206
                if resumed and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                    raise ValueError('Invalid Content-Range; refusing append')
                size=offset if resumed else 0;last_print=time.monotonic()
                with part.open('ab' if resumed else 'wb') as f:
                    while block:=response.read(1024*1024):
                        size+=len(block)
                        if size>600*1024*1024:raise ValueError('Weight exceeds 600 MiB limit')
                        f.write(block)
                        if time.monotonic()-start>timeout:raise TimeoutError('Download step deadline reached; partial bytes retained')
                        if time.monotonic()-last_print>20:
                            print(f'{dest.name}: {size/1024/1024:.1f} MiB received',flush=True);last_print=time.monotonic()
            if digest(part)!=expected:
                part.unlink();raise ValueError(f'SHA256 mismatch for {dest.name}; corrupt partial discarded')
            part.replace(dest);return
        except (OSError,urllib.error.URLError,ValueError) as exc:
            last_error=exc
            if time.monotonic()-start>=timeout:break
            print(f'{dest.name}: attempt {attempt+1} failed ({type(exc).__name__}); retaining valid partial bytes',flush=True)
    raise RuntimeError(f'Download incomplete: {dest.name}; rerun to resume') from last_error


def command(args, *, timeout):
    # A timed-out shell must not leave a child installer downloading indefinitely.
    p=subprocess.Popen([str(x) for x in args],cwd=ROOT,start_new_session=True)
    try:
        code=p.wait(timeout=timeout)
        if code:raise RuntimeError(f'Command failed with exit {code}: {args[0]}')
    except BaseException:
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        raise


class Timings:
    def __init__(self,mode):
        self.start=time.monotonic();self.data={'mode':mode,'clip_commit':CLIP_COMMIT,'steps':[],'status':'running'}
        self.path=ROOT/'.runtime/setup_timings'/f'{time.strftime("%Y%m%dT%H%M%S")}_{mode}_{uuid.uuid4().hex[:8]}.json'
        self.path.parent.mkdir(parents=True,exist_ok=True)
    def save(self):
        self.data['elapsed_s']=round(time.monotonic()-self.start,3)
        self.path.write_text(json.dumps(self.data,indent=2)+'\n')
    @contextmanager
    def step(self,name):
        print(f'\n== {name} ==',flush=True);start=time.monotonic();row={'step':name,'status':'running'};self.data['steps'].append(row);self.save()
        try:yield
        except BaseException:
            row['status']='failed';self.data['status']='failed';raise
        else:row['status']='ok'
        finally:
            row['elapsed_s']=round(time.monotonic()-start,3);self.save();print(f'TIMER {name}: {row["elapsed_s"]:.3f}s [{row["status"]}]',flush=True)


def settings(path,updates):
    """Replace only owned model settings, preserve every unrelated line/key."""
    lines=path.read_text().splitlines() if path.exists() else [];seen=set();out=[]
    for line in lines:
        key=line.split('=',1)[0].strip()
        if key in updates:
            if key not in seen:out.append(f'{key}={updates[key]}');seen.add(key)
        else:out.append(line)
    out.extend(f'{k}={v}' for k,v in updates.items() if k not in seen)
    # Paths are JSON quoted, which python-dotenv accepts even with spaces.
    temporary=path.with_name(path.name+'.setup.part');temporary.write_text('\n'.join(out)+'\n');temporary.chmod(0o600);temporary.replace(path)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--minimal',action='store_true',help='Core app + RapidOCR only; detection unavailable/disabled, no YOLO/CLIP/Florence')
    ap.add_argument('--step-timeout',type=int,default=900,help='Bound each installer/download step; interrupted downloads resume on rerun (default 900 seconds)')
    args=ap.parse_args(argv)
    if not 1<=args.step_timeout<=3600:ap.error('--step-timeout must be 1..3600 seconds')
    os.chdir(ROOT);rt=ROOT/'.runtime';py=ROOT/'.venv/bin/python';runtime=rt/'venv/bin/python';limit=args.step_timeout
    timer=Timings('minimal' if args.minimal else 'full')
    def run(*cmd):command(cmd,timeout=limit)
    try:
        with timer.step('core app'):
            if not py.exists():run('uv','venv','--python','3.12',ROOT/'.venv')
            run('uv','pip','install','-p',py,'-c','requirements/setup-constraints.txt','-e','.[dev]')
        with timer.step('RapidOCR packages'):
            # Avoid RapidOCR's non-headless OpenCV dependency shadowing the app wheel.
            run('uv','pip','install','-p',py,'--no-deps','rapidocr==3.9.2','onnxruntime==1.30.0')
            run('uv','pip','install','-p',py,'-c','requirements/setup-constraints.txt','pyclipper','shapely','omegaconf','colorlog','pyyaml','six','tqdm','pillow','requests')
        with timer.step('RapidOCR Latin models + warmup'):
            directory=subprocess.check_output([str(py),'-c','import rapidocr,pathlib;print(pathlib.Path(rapidocr.__file__).parent/"models")'],text=True,cwd=ROOT,timeout=30).strip()
            for relative,sha in RAPID_MODELS:fetch(RAPID_BASE+relative,Path(directory)/Path(relative).name,sha,timeout=limit)
            run(py,'-c','from providers.rapidocr_reader import RapidOCRAdapter; from pathlib import Path; RapidOCRAdapter(Path(".runtime/rapidocr_warmup")).close(); print("RapidOCR Latin ready")')
        if not args.minimal:
            with timer.step('YOLO/CLIP runtime (pinned)'):
                if not runtime.exists():run('uv','venv','--python','3.12',rt/'venv')
                run('uv','pip','install','-p',runtime,'--build-constraint','requirements/setup-build-constraints.txt','setuptools==80.9.0','torch==2.14.1','torchvision==0.29.1','ultralytics==8.4.0','ftfy==6.3.1','regex==2026.9.29','tqdm==4.70.1','pillow==12.3.0','numpy==2.5.3','opencv-python-headless==5.0.0.93',f'clip @ git+https://github.com/openai/CLIP.git@{CLIP_COMMIT}')
            with timer.step('YOLO-World weights'):
                fetch('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-worldv2.pt',rt/'models/yolov8m-worldv2.pt','b614d33aa35b8e61d988041ff6939dfb3ed627af88ccaf643e4cdb822eb41d71',timeout=limit)
            with timer.step('CLIP weights'):
                sha='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
                fetch(f'https://openaipublic.azureedge.net/clip/models/{sha}/ViT-B-32.pt',rt/'models/clip/ViT-B-32.pt',sha,timeout=limit)
            with timer.step('spine OBB weights'):
                target=rt/'models/book-spine-best.pt';sha='e5a4cc93fed9904d71a533bce2310e5705b08815d8d11869768b0c1de5b3a1ce';target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists() or digest(target)!=sha:
                    part=target.with_name(target.name+'.part')
                    run('uvx','--from','gdown==5.2.0','gdown','--continue','1zIxEQuPFWKhE7PxC4Ry5vJEnAI1MWOFB','-O',part)
                    if digest(part)!=sha:part.unlink();raise ValueError('Spine OBB checksum mismatch')
                    part.replace(target)
                else:print('cached: spine OBB',flush=True)
            with timer.step('EasyOCR + CRAFT'):
                for url,name,archive_sha,model,model_sha in [
                  ('https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/latin_g2.zip','latin_g2.zip','29f1920c493378da65a59793fb70e7e190504662b6bed57ab26f4067eb5f3769','latin_g2.pth','aaa95be1c4a9cb3496879bed7c520886ce1164f89e026f0c54488394e74e8c55'),
                  ('https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip','craft_mlt_25k.zip','8dc6a1c703a89ed56308ef742d26ebd45c656248cbbbda6e7fe60e569f873e65','craft_mlt_25k.pth','4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17')]:
                    dest=rt/'easyocr_models'/model
                    if not dest.exists() or digest(dest)!=model_sha:
                        archive=dest.parent/name;fetch(url,archive,archive_sha,timeout=limit)
                        with zipfile.ZipFile(archive) as z:dest.write_bytes(z.read(model))
                        if digest(dest)!=model_sha:raise ValueError('Extracted OCR model checksum mismatch')
                run('uv','pip','install','-p',runtime,'--target',rt/'easyocr_packages','--no-deps','easyocr==1.7.2','python-bidi==0.6.11','pyclipper==1.4.0','scikit-image==0.26.0','imageio==2.38.0','lazy-loader==0.6','tifffile==2026.9.20','ninja==1.13.2')
            with timer.step('detector model warmup'):
                run(runtime,'-c','import os,sys; os.environ["YOLO_CONFIG_DIR"]=sys.argv[2]; os.environ["YOLO_AUTOINSTALL"]="false"; from providers.local_detector import LocalDetector; LocalDetector(sys.argv[1]); print("Local detector weights/runtime ready")',rt/'models',rt/'setup_detector_warmup')
        with timer.step('local-only settings'):
            updates={'LIBRARY_DETECTOR':'none' if args.minimal else 'local','LIBRARY_READER':'rapidocr'}
            if not args.minimal:
                updates.update({k:json.dumps(str(v)) for k,v in {'LIBRARY_DETECTOR_PYTHON':runtime,'LIBRARY_DETECTOR_WEIGHTS':rt/'models','LIBRARY_OCR_MODEL_DIR':rt/'easyocr_models','LIBRARY_OCR_PACKAGE_DIR':rt/'easyocr_packages'}.items()})
            settings(ROOT/'.env',updates)
        timer.data['status']='ok';timer.save();print(f'COMPLETE {timer.data["mode"]}: {timer.data["elapsed_s"]:.3f}s; timings: {timer.path.relative_to(ROOT)}',flush=True)
        return 0
    except (Exception,KeyboardInterrupt) as exc:
        timer.data['status']='failed';timer.save();print(f'SETUP FAILED ({type(exc).__name__}): {exc}\nRerun the same command to resume. Timings: {timer.path.relative_to(ROOT)}',flush=True);return 1


if __name__=='__main__':raise SystemExit(main())
