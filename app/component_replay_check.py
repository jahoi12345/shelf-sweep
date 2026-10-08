"""REAL local detector/OCR WebSocket replay through FastAPI; no camera/mic/voice.

Run: .venv/bin/python -m app.component_replay_check --out runs/frontend_e2e_v1
Uses TestClient in-process because this sandbox cannot bind a new test port.
Transport is ASGI TestClient; model adapters/stages, source frames and downloaded
ZIP bytes are real. Source recording is untouched. No human approval is invoked.
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import os
from pathlib import Path
import time
import zipfile
from fastapi.testclient import TestClient
from app.replay import iter_video_frames
from pipeline import contracts as C


def run(out):
    root=Path(out).resolve()
    if root.exists():raise ValueError('Choose a new versioned output directory')
    root.mkdir(parents=True)
    os.environ.update(LIBRARY_DETECTOR='local',LIBRARY_READER='easyocr',LIBRARY_OCR_MAX_BOOKS='4',LIBRARY_RUNS_DIR=str(root/'sessions'))
    from app import server
    server.RUNS_DIR=root/'sessions'
    start=time.perf_counter();events=[]
    with TestClient(server.app,base_url='http://127.0.0.1:4179',headers={'origin':'http://127.0.0.1:4179'}) as client:
        created=client.post('/api/sessions');created.raise_for_status();sid=created.json()['session_id']
        client.get('/annotations').raise_for_status()
        payload=client.get('/annotations/annotation_payload.json').json()
        assert payload['annotations']['provenance']['review_status']=='pending'
        client.get('/annotations/evidence/f00000.jpg').raise_for_status()
        assert client.get('/annotations/evidence/not-listed.jpg').status_code==404
        with client.websocket_connect('ws://127.0.0.1:4179/ws/live?session_id='+sid,headers={'origin':'http://127.0.0.1:4179'}) as ws:
            while True:
                event=ws.receive_json();events.append(event)
                if event['type']=='ready':assert event['voice'] is False;break
            for t,jpeg in iter_video_frames('sweeps/IMG_2281.MOV',fps=2):
                # Use actual source video clocks; pause only for the WebSocket rate limit.
                ws.send_json({'type':'evidence_frame','source':'replay','t_s':t,'data':base64.b64encode(jpeg).decode()})
                while True:
                    event=ws.receive_json();events.append(event)
                    if event['type']=='frame':break
                time.sleep(.18)
            ws.send_json({'type':'end_sweep'})
            while True:
                event=ws.receive_json();events.append(event)
                if event['type']=='state' and event['state']['status']=='finished':break
            final=event['state']
            assert final['totals']['book_count']>0 and final['totals']['item_count']>0
            assert any(b.get('visible_text') for b in final['books'])
            response=client.get('/api/sessions/'+sid+'/packet');response.raise_for_status()
            (root/'packet.zip').write_bytes(response.content)
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                assert archive.testzip() is None
                assert len(archive.namelist())==len(set(archive.namelist())), 'duplicate ZIP members'
                assert 'readings.json' in archive.namelist() and any(n.startswith('ocr_responses/') for n in archive.namelist())
            C.write_json(root/'final_state.json',final)
        # No automation interacts with human annotation approval controls.
        C.write_json(root/'verification.json',{'runtime':'real local YOLO + EasyOCR; ASGI TestClient transport',
           'device_checks':'camera/mic/voice not exercised','session_id':sid,'elapsed_s':round(time.perf_counter()-start,3),
           'state_events':sum(e['type']=='state' for e in events),'frame_events':sum(e['type']=='frame' for e in events),
           'provisional_events_with_detections':sum(e['type']=='state' and e['state']['status']=='capturing' and
              bool(e['state'].get('books') or e['state'].get('provisional_unmerged')) for e in events),
           'literal_text_lines':sum(bool(b.get('visible_text')) for b in final['books']),
           'totals':final['totals'],'human_annotation_status':'pending; no approvals created','zip_valid':True})
    print(root/'verification.json')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True);run(parser.parse_args().out)
