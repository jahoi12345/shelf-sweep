"""Optional open-source Latin CRNN OCR for already detected spine crops.

This is a common text fallback, not an upstream detector's complete workflow.
Initialize once. Dependencies/model setup belongs to the app runtime; comparison uses
an isolated runtime. Only existing pretrained latin_g2 weights, no auto download.
"""
from __future__ import annotations
import asyncio,json,threading,time
import uuid
from pathlib import Path
from providers.local_ocr import digest,reading_payload,rotation_to_crop

class EasyOCRAdapter:
    def __init__(self,model_dir,*,evidence_dir,rotations=(0,90,180,270)):
        import easyocr,torch
        self.model_dir=Path(model_dir).resolve();self.evidence_dir=Path(evidence_dir).resolve()
        self.evidence_dir.mkdir(parents=True,exist_ok=True)
        self.rotations=tuple(rotations);self._lock=threading.Lock()
        if not self.rotations or any(x not in (0,90,180,270) for x in self.rotations):raise ValueError('Invalid OCR rotations')
        torch.set_num_threads(4);start=time.perf_counter()
        self.craft=(self.model_dir/'craft_mlt_25k.pth').is_file()   # with the CRAFT text detector present, find text boxes first (recognising the whole crop as one line mostly yields '[?]')
        self.reader=easyocr.Reader(['es','en'],gpu=False,detector=self.craft,model_storage_directory=str(self.model_dir),
                                  user_network_directory=str(self.model_dir/'user_network'),download_enabled=False,verbose=False)
        self.setup_s=time.perf_counter()-start
        self.runtime={'engine':'EasyOCR','version':easyocr.__version__,'torch':torch.__version__,
                      'device':'cpu','languages':['es','en'],'model':'latin_g2.pth','adapter_sha256':digest(__file__),
                      'model_sha256':digest(self.model_dir/'latin_g2.pth'),'source_url':'https://github.com/JaidedAI/EasyOCR',
                      'license':'Apache-2.0 (EasyOCR package)','text_detection':'CRAFT (craft_mlt_25k.pth) + row merge' if self.craft else False,
                      'decoder':'greedy; no title dictionary or catalogue words',
                      'model_preprocessing':'CRNN internally resizes native input to model height; saved evidence/crops remain native'}
    async def __call__(self,*,image_paths=None,image_path=None,task='read',config=None,**kwargs):
        if task!='read':raise ValueError('EasyOCRAdapter is never a book detector')
        paths=image_paths if image_paths is not None else ([image_path] if image_path else [])
        if len(paths)!=1:raise ValueError('Requires one native crop')
        return await asyncio.to_thread(self._infer,Path(paths[0]).resolve(),config or {})
    def _infer(self,path,config):
        import cv2,hashlib
        with self._lock:
            start=time.perf_counter();image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            if image is None:raise ValueError('Cannot open crop')
            h,w=image.shape;variants=[]
            for rotation in self.rotations:
                oriented=image if rotation==0 else cv2.rotate(image,{90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}[rotation])
                oh,ow=oriented.shape;t=time.perf_counter()
                if self.craft:
                    from providers.rapidocr_reader import _merge_rows
                    result=self.reader.readtext(oriented,detail=1,paragraph=False,decoder='greedy',batch_size=1,workers=0)
                    rows=_merge_rows([b for b,t,c in result],[t for b,t,c in result],[c for b,t,c in result])
                    lines=[{'bbox_oriented_pixels':None,'candidates':[{'text':tx,'confidence':float(cf)}]} for tx,cf,_ in rows]
                else:
                    result=self.reader.recognize(oriented,horizontal_list=[[0,ow,0,oh]],free_list=[],detail=1,
                                                 decoder='greedy',batch_size=1,workers=0,paragraph=False)
                    lines=[{'bbox_oriented_pixels':[[int(px),int(py)] for px,py in box],
                            'candidates':[{'text':text,'confidence':float(confidence)}]} for box,text,confidence in result]
                variants.append({'rotation_deg_clockwise':rotation,'lines':lines,'latency_s':time.perf_counter()-t,
                                 'oriented_width_px':ow,'oriented_height_px':oh,
                                 'oriented_pixels_to_native_crop_3x3':rotation_to_crop(rotation,w,h)})
            payload,rotation,uncertain=reading_payload(variants)
            key=hashlib.sha256(json.dumps({'crop':digest(path),'model':self.runtime['model_sha256'],'rotations':self.rotations,
                              'turn':config.get('source_turn_id'),'revision':config.get('session_revision')},sort_keys=True).encode()).hexdigest()
            target=self.evidence_dir/(key+'.'+uuid.uuid4().hex+'.json')
            record={'runtime':self.runtime,'input_path':str(path),'input_sha256':digest(path),'variants':variants,
                    'selected_rotation_deg_clockwise':rotation,'uncertain_line_indices':uncertain,'data':payload,
                    'confidence_note':'OCR confidence is not calibrated identity or correctness',
                    'source_turn_id':config.get('source_turn_id'),'session_revision':config.get('session_revision')}
            target.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
            return {'data':payload,'request_ref':str(target),'ocr_evidence':{'path':str(target),'sha256':digest(target),
                    'selected_rotation_deg_clockwise':rotation,'oriented_pixels_to_native_crop_3x3':rotation_to_crop(rotation,w,h)},
                    'metrics':{'provider':'local_easyocr','model':'latin_g2','request_count':0,'local_inference_count':len(variants),
                               'latency_s':time.perf_counter()-start,'actual_cost':0,'cost_provenance':'Local CPU OCR; no API calls','cache_hit':False}}
    def close(self):pass
