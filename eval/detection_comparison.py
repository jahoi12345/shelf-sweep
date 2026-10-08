"""Bounded, credential-free upstream detection-stage comparison on common images.

Only explicitly audited entrypoints are executed. Partial detection execution is
never labelled a completed OCR reader test. Human metrics use eval.score, not
model outputs as labels. Run in the isolated comparison venv (see handoff).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pipeline import contracts as C
from pipeline.detect import read_json, save_json, native_box

REPO = Path(__file__).resolve().parents[1]
SOURCES = {
    "book_spine_scanner": ("eval/spike/book_spine_scanner", "book_spine_scanner.py", "yolo11n.pt"),
    "spinal_chord": ("eval/spike/spinal-chord", "bookshelf/detect.py", "yolov8m-worldv2.pt"),
    "book_spine_recognition": ("eval/spike/book-spine-recognition", "backend/python-scripts/detect.py", "book-spine-best.pt")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_record(name):
    directory, file, weights = SOURCES[name]
    source = REPO / directory / file
    commit = subprocess.check_output(["git", "-C", str(REPO/directory), "rev-parse", "HEAD"], text=True).strip()
    return {"source_path":directory+"/"+file,"source_commit":commit,"source_sha256":sha(source),
        "weights_file":weights,"audit":"Read imports and called entrypoints; never execute constructors that load hosted OCR credentials"}


def module_from_path(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run_backend(name,inputs_path,out):
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from PIL import Image
    torch.set_num_threads(4)
    out = Path(out).resolve()
    inputs = read_json(inputs_path)
    source_root = (Path(inputs_path).resolve().parent / inputs["source_run"]).resolve()
    manifest = read_json(source_root/"manifest.json")
    out.mkdir(parents=True,exist_ok=True)
    if (out/"candidates.json").exists():
        raise ValueError("Refusing to replace published candidate version")
    # Preserve the whole-video namespace and exact evidence; only common input
    # frames are inferred, and omitted selected views remain explicit failures.
    for frame in manifest["data"]["frames"]:
        dest=out/frame["image_path"]
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source_root/frame["image_path"],dest)
    manifest["run_id"] = out.name
    save_json(out/"manifest.json",manifest)
    meta=source_record(name)
    directory,file,weights_name=SOURCES[name]
    source=REPO/directory/file
    weights=Path(inputs_path).resolve().parent/"models"/weights_name
    meta["weights_sha256"]=sha(weights)
    setup_start=time.perf_counter()
    # All model files are explicit local files; automatic dependency installs are disabled.
    if name == "book_spine_scanner":
        upstream=module_from_path("audited_spine_scanner",source)
        model=upstream.BookSpineScanner(model_weights=str(weights))
        reading_status="not_provided_by_upstream"
        adaptation="Unmodified audited BookSpineScanner.process_bookshelf; default conf .35, includes cell-phone class as book upstream"
    elif name == "spinal_chord":
        import ultralytics.utils
        ultralytics.utils.WEIGHTS_DIR=weights.parent
        upstream=module_from_path("audited_spinal_detection",source)
        model=upstream._get_model(str(weights))
        reading_status="not_executed_local_Qwen_weights_and_runtime_not_prepared"
        adaptation="Audited detect_books with upstream prompts/conf .2 and geometry filtering, local pretrained weights; no hosted reader invoked"
    else:
        # Execute precisely the audited detection/crop method, not the API/CUDA
        # constructor. This is explicitly an adapted component test, not E2E.
        tree=ast.parse(source.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="BookDetector")
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="detect_and_crop_books")
        method.returns=None
        for arg in method.args.args:
            arg.annotation=None
        node=ast.Module(body=[method],type_ignores=[])
        ast.fix_missing_locations(node)
        namespace={"np":np,"cv2":cv2}
        exec(compile(node,str(source),"exec"),namespace)
        raw_model=ultralytics.YOLO(str(weights)).to("cpu")
        class Recorder:
            latest=None
            def predict(self,**kwargs):
                self.latest=raw_model.predict(**kwargs)
                return self.latest
        class Context:
            pass
        model=Context();model.model=Recorder()
        crop_method=namespace["detect_and_crop_books"]
        reading_status="not_executed_Google_Cloud_Vision_requires_distinct_credentials"
        adaptation="Audited upstream OBB detection/crop method; bypasses hardcoded CUDA/API constructor; supplied public custom weights; no super-resolution or OCR"
    setup_s=time.perf_counter()-setup_start
    observations,frame_metrics,issues=[],[],[]
    for frame in inputs["frames"]:
        fid=frame["frame_id"]
        image_path=out/frame["image_path"]
        if sha(image_path)!=frame["sha256"]:
            raise ValueError("Common evidence byte mismatch")
        frame_dir=out/"raw"/fid
        frame_dir.mkdir(parents=True,exist_ok=True)
        t=time.perf_counter()
        image=cv2.imread(str(image_path))
        if name == "book_spine_scanner":
            result=model.process_bookshelf(str(image_path),output_dir=str(frame_dir),show_display=False)
            boxes=[([b["position"][k] for k in ["x1","y1","x2","y2"]],b["confidence"],None) for b in result["books"]]
        elif name == "spinal_chord":
            result=upstream.detect_books(Image.open(image_path),weights=str(weights))
            upstream.save_crops(result,frame_dir/"crops")
            boxes=[(list(d.bbox),d.confidence,None) for d in result]
        else:
            result=crop_method(model,image)
            raw_result=model.model.latest[0]
            raw=[]
            if raw_result.obb is not None:
                for box in raw_result.obb:
                    points=box.xyxyxyxy.cpu().numpy().reshape(-1,2)
                    raw.append({"polygon_xy":points.tolist(),"confidence":float(box.conf[0])})
            save_json(frame_dir/"raw_obb.json",raw)
            boxes=[]
            for index,(crop,flat) in enumerate(result):
                polygon=np.array(flat).reshape(-1,2)
                matching=min(raw,key=lambda r:float(np.sum((np.array(r["polygon_xy"])-polygon)**2))) if raw else {}
                conf=matching.get("confidence")
                box=[float(polygon[:,0].min()),float(polygon[:,1].min()),float(polygon[:,0].max()),float(polygon[:,1].max())]
                boxes.append((box,conf,polygon.tolist()))
                cv2.imwrite(str(frame_dir/f"crop_{index:03d}.png"),crop)
        elapsed=time.perf_counter()-t
        for index,(box,confidence,polygon) in enumerate(boxes):
            # Upstream rounding can touch edges; preserve raw source in raw artifact.
            w,h=frame["width_px"],frame["height_px"]
            box=[max(0,box[0]),max(0,box[1]),min(w,box[2]),min(h,box[3])]
            box=native_box(box,"xyxy_pixels",[0,0,w,h],w,h)
            o={"observation_id":C.observation_id(fid,index),"frame_id":fid,"bbox_xyxy":box,
                "kind":"book","category":"book","orientation":"unknown","confidence":confidence,
                "occlusion":"unknown","shelf_hint":None,"evidence_notes":"Upstream detector prediction, not human truth; unreadable identity unknown",
                "issues":["Book-only upstream detection; non-book objects not inventoried"],"response_ref":f"raw/{fid}/detections.json"}
            if polygon:
                o["polygon_xy"]=[[min(w,max(0,p[0])),min(h,max(0,p[1]))] for p in polygon]
            observations.append(o)
        save_json(frame_dir/"detections.json",[o for o in observations if o["frame_id"]==fid])
        frame_metrics.append({"frame_id":fid,"elapsed_s":elapsed,"candidate_count":len(boxes),"input_sha256":frame["sha256"]})
    inferred={f["frame_id"] for f in inputs["frames"]}
    coverage=[{"frame_id":f["frame_id"],"status":"processed" if f["frame_id"] in inferred else "unprocessed" if f["selected"] else "not_selected"}
        for f in manifest["data"]["frames"]]
    issues=[{"reason":"Bounded two-frame detector-stage test only; non-book inventory and title reading not executed"}]
    candidates=C.envelope(out.name,"detect",{"observations":observations,"frames":coverage},status="partial",issues=issues,
        metrics={"provider":"local_pretrained","model":weights_name,"setup_s":setup_s,"elapsed_s":sum(f["elapsed_s"] for f in frame_metrics),
            "request_count":0,"actual_cost":0,"cost_provenance":"Local CPU inference; no API requests","frame_metrics":frame_metrics})
    errors=C.validate_candidates(candidates,manifest)
    if errors:
        raise ValueError("Invalid upstream coordinate output: "+"; ".join(errors))
    save_json(out/"candidates.json",candidates)
    report={"candidate":name,"execution_status":"executed_detection_component","full_reader_executed":False,
        "reading_status":reading_status,"source":meta,"adaptation":adaptation,
        "setup_s":setup_s,"frames":frame_metrics,"runtime":{"python":sys.version.split()[0],"torch":torch.__version__,"ultralytics":ultralytics.__version__,"device":"cpu","threads":4},
        "accuracy":{"status":"unavailable","reason":"Frozen human labels absent","clutter_false_positives":None,"missed_books":None,"title_correctness":None},
        "API_requests":0,"API_cost":0,"bilingual_status":"Book shapes are language-agnostic; Spanish/English title reading is unmeasured"}
    save_json(out/"execution.json",report)
    return report


def run(name,inputs_path,out,python):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    # Never pass API credentials or auth sockets to checked-out code.
    env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ["KEY","TOKEN","SECRET","CREDENTIAL","PASSWORD","AUTH"])}
    root=Path(inputs_path).resolve().parent
    env.update(YOLO_CONFIG_DIR=str(root/"yolo_config"),MPLCONFIGDIR=str(root/"mpl_config"),
        YOLO_AUTOINSTALL="false",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    for p in [root/"yolo_config"/"Ultralytics",root/"mpl_config"]:
        p.mkdir(parents=True,exist_ok=True)
    command=[python,"-m","eval.detection_comparison","--candidate",name,"--inputs",str(Path(inputs_path).resolve()),"--out",str(out),"--backend"]
    t=time.perf_counter()
    result=subprocess.run(command,cwd=REPO,env=env,capture_output=True,text=True,timeout=180)
    (out/"execution_stdout.log").write_text(result.stdout)
    (out/"execution_stderr.log").write_text(result.stderr)
    if result.returncode:
        report={"candidate":name,"execution_status":"attempted_but_failed","full_reader_executed":False,
            "exit_code":result.returncode,"wall_s":time.perf_counter()-t,"reason":"See sanitized credential-free subprocess logs; inference may not have completed"}
        save_json(out/"attempt.json",report)
        return report
    report=read_json(out/"execution.json")
    report["process_wall_s"]=time.perf_counter()-t
    save_json(out/"execution.json",report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate",choices=sorted(SOURCES),required=True)
    p.add_argument("--inputs",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--python",default=sys.executable)
    p.add_argument("--backend",action="store_true",help=argparse.SUPPRESS)
    args=p.parse_args()
    result=_run_backend(args.candidate,args.inputs,args.out) if args.backend else run(args.candidate,args.inputs,args.out,args.python)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
