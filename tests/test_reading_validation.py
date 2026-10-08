"""Independent REVIEW specifications; red tests document unfixed bugs.

All small data fixtures below are synthetic contract probes, never human truth.
Real-run checks read existing artifacts only. No inference, network or approvals.
"""
import asyncio
import copy
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import group_reader as GR, score_titles as ST
from eval.score import _title_counts, annotation_problems, match_boxes
from pipeline.read_spines import native_crop
from pipeline.resolve import normalize_title
from providers import readers

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/final_candidate_yt1_v2'
DEV = ROOT / 'eval/dev_labels/model_read_yt1_v1.json'


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def toy_run(text='Dune Messiah', confidence=.9, complete=True):
    return {'cands':[{'observation_id':'o','frame_id':'f','kind':'book','bbox_xyxy':[0,0,10,10]}],
            'frames':{'f':{'timestamp_s':0}},'candidates_sha256':'candidate-a','scan_sha':'scan-a',
            'readings':{'o':{'title_text':text,'read_confidence':confidence,'title_complete':complete}},
            'res':{},'run_id':'synthetic-contract-probe'}


def toy_row(title='Dune'):
    return {'observation_id':'o','truth_title':title,'language':'en','label_run':'synthetic-contract-probe',
            'candidates_sha256':'candidate-a','label_scan_sha':'scan-a','timestamp_s':0,'bbox_xyxy':[0,0,10,10]}


@pytest.fixture
def resolver():
    """The three resolver regressions below now run against the PRODUCTION fragment resolver (pipeline.resolve.evaluate_candidates, rule v6, ported from the reader benchmark).
    The adapter keeps the old evaluate(text, candidates, rule, cov_min) call shape; production clamps coverage to its >= 0.7 floor."""
    from types import SimpleNamespace
    from pipeline import resolve as production
    return SimpleNamespace(evaluate=lambda text, cands, rule='v6', cov_min=.7: production.evaluate_candidates(text, cands, cov_min))


def test_control_unicode_normalization_and_absent_truth():
    assert ST.compare('Cien años de soledad','CIEN AÑOS DE SOLEDAD')['exact']
    assert not ST.compare('Cien años de soledad','Cien anos de soledad')['exact']
    assert normalize_title('García') != normalize_title('Garcia')
    d=ST.build([],Path('unused-no-labels'))
    assert d['status']=='unavailable'
    assert d['layers']['reading']['en']['exact_title']['value'] is None


def test_regression_wrong_full_title_cannot_escape_via_containment():
    R=toy_run();got=ST.score([toy_row()],R,{})['layers']['reading']['en']['counts']
    expected=_title_counts({'o':['Dune']},{'o':{'title':'Dune Messiah','confidence':.9}},.85)
    assert expected['confident_wrong']==1
    assert got['confidently_wrong']==expected['confident_wrong']


def test_regression_confidence_threshold_matches_declared_085():
    got=ST.score([toy_row()],toy_run('Unrelated novel',.82),{})['layers']['reading']['en']['counts']
    expected=_title_counts({'o':['Dune']},{'o':{'title':'Unrelated novel','confidence':.82}},.85)
    assert got['confidently_wrong']==expected['confident_wrong']


def test_regression_ocr_block_is_not_an_asserted_complete_title():
    R=toy_run('Dune',None,False)
    R['readings']['o']={'visible_text':'Dune','title_text':None,'title_complete':False,'read_confidence':None}
    got=ST.score([toy_row()],R,{})['layers']['reading']['en']['counts']
    expected=_title_counts({'o':['Dune']},{'o':{'title':None,'confidence':None}},.85)
    assert expected['abstained']==1
    assert got['exact']==expected['correct']


def test_regression_containment_requires_token_boundaries_and_full_truth():
    assert not ST.compare('Dune','Sandune')['contained']


def test_regression_pending_model_labels_cannot_be_scored_as_human_truth(tmp_path,monkeypatch):
    p=tmp_path/'synthetic_nontruth.json'
    d={'fixture':True,'purpose':'evaluation_truth','kind':'title_labels','run_id':'synthetic-contract-probe',
       'annotator':None,'provenance':{'source_type':'model_read','review_status':'pending','frozen_at':None},
       'scan':{'sha256':'scan-a'},'source_hashes':{'candidates_sha256':'candidate-a'},
       'sample':[toy_row()],'labels':{'o':{'status':'labeled','title':'Dune','language':'en'}}}
    p.write_text(json.dumps(d))
    assert annotation_problems({'annotation_version':1,'provenance':d['provenance'],'frames':[]})
    monkeypatch.setattr(ST,'load_run',lambda _:toy_run('Dune'))
    try:
        result=ST.build([p],Path('unused'))
    except ValueError:
        return
    assert result['status']=='unavailable'


def test_regression_unknown_scan_hash_cannot_establish_cross_run_identity():
    row=toy_row();row.update(candidates_sha256='different-candidates',label_scan_sha=None)
    R=toy_run();R['scan_sha']=None
    assert ST.match([row],R)[ST.truth_key(row)] is None   # match() is keyed by artifact+observation id, not the bare id


def test_regression_matching_keeps_maximum_cardinality():
    truth_boxes=[[0,0,10,10],[3,0,13,10]];pred_boxes=[[1,0,11,10],[-2,0,8,10]]
    rows=[{**toy_row(),'observation_id':f't{i}','candidates_sha256':'other','bbox_xyxy':b} for i,b in enumerate(truth_boxes)]
    R=toy_run();R['cands']=[{'observation_id':f'p{i}','frame_id':'f','bbox_xyxy':b,'kind':'book'} for i,b in enumerate(pred_boxes)]
    got=ST.match(rows,R)
    expected=match_boxes([{'bbox_xyxy':b} for b in truth_boxes],R['cands'],.5)
    assert len(expected)==2
    assert sum(v is not None for v in got.values())==len(expected)


def test_regression_usable_group_sighting_not_hidden_by_narrow_sharp_sighting():
    frames={'a':{'quality':{'sharpness':1000},'width_px':200,'height_px':200},
            'b':{'quality':{'sharpness':10},'width_px':200,'height_px':200}}
    obs={'narrow':{'observation_id':'narrow','frame_id':'a','bbox_xyxy':[30,30,40,130]},
         'wide':{'observation_id':'wide','frame_id':'b','bbox_xyxy':[30,30,60,130]}}
    image=np.random.default_rng(1).integers(0,256,(200,200),dtype=np.uint8)
    plan=GR.plan([{'instance_id':'proposed-g','observation_ids':['narrow','wide']}],obs,frames,lambda _:image)
    assert plan['groups'][0]['status']=='attempt'


def test_regression_all_rotation_attempts_are_linked_in_pool_response(tmp_path):
    class FakeAdapter:
        async def __call__(self,**kwargs):
            first=self.rotations==(90,270)
            path='first.json' if first else 'second.json'
            return {'data':{'visible_text':None if first else 'Literal text'},'request_ref':path,'ocr_evidence':{'path':path},
                    'metrics':{'local_inference_count':2,'latency_s':.1}}
    pool=readers._RapidPool(tmp_path,1);adapter=FakeAdapter();pool._acquire=lambda:adapter
    out=asyncio.run(pool(task='read',image_paths=['unused']))
    def strings(x):
        if isinstance(x,dict):return {s for v in x.values() for s in strings(v)}
        if isinstance(x,list):return {s for v in x for s in strings(v)}
        return {x} if isinstance(x,str) else set()
    assert {'first.json','second.json'} <= strings(out)


def test_regression_spanish_text_cannot_resolve_to_unseen_english_title(resolver):
    text='Harry Potter y el prisionero de Azkaban\nJ. K. Rowling'
    cands=[{'work':'/works/OL1W','titles':['Harry Potter and the Prisoner of Azkaban'],'authors':['J. K. Rowling'],'source':'synthetic'}]
    decision,result=resolver.evaluate(text,cands,'v6',.6)
    assert decision!='accept',result


def test_regression_catalogue_must_not_complete_unseen_title_tokens(resolver):
    cands=[{'work':'/works/OL1W','titles':['Alpha Beta Gamma'],'authors':['Anne Miller'],'source':'synthetic'}]
    decision,result=resolver.evaluate('Alpha Beta\nAnne Miller',cands,'v6',.6)
    assert decision!='accept',result


def test_regression_distinct_author_work_contradiction_must_abstain(resolver):
    cands=[{'work':'/works/OL1W','titles':['Dune Messiah'],'authors':['John Herbert'],'source':'synthetic'},
           {'work':'/works/OL2W','titles':['Dune Messiah'],'authors':['Frank Herbert'],'source':'synthetic'}]
    decision,result=resolver.evaluate('Dune Messiah\nFrank Herbert',cands,'v6',.6)
    assert decision!='accept' or result['work']=='/works/OL2W',result


def test_control_dev_smoke_is_nontruth_with_no_language_denominators():
    if not DEV.exists() or not (RUN/'readings.json').exists():pytest.skip('real artifacts absent')
    doc=ST.build([],RUN,dev_smoke=DEV)
    assert doc['kind']=='title_score_DEV_SMOKE' and doc['status']=='dev_smoke'
    assert doc['truth']['dev_smoke'] and 'NOT human truth' in doc['truth']['warning']
    assert doc['layers']['reading']['all']['counts']['n']==36
    for language in ('es','en'):assert not doc['layers']['reading'][language]['exact_title']['available']


def test_control_saved_native_crop_hashes_and_pixels_intact():
    if not (RUN/'readings.json').exists():pytest.skip('real artifacts absent')
    readings=json.loads((RUN/'readings.json').read_text())['data']['readings']
    frames={f['frame_id']:f for f in json.loads((RUN/'manifest.json').read_text())['data']['frames']}
    obs={o['observation_id']:o for o in json.loads((RUN/'candidates.json').read_text())['data']['observations']}
    cache={}
    for r in readings:
        crop=RUN/r['crop_path'];raw_path=Path(r['ocr_evidence']['path']);raw=json.loads(raw_path.read_text())
        assert digest(crop)==r['provenance']['crop_sha256']==raw['input_sha256']
        assert digest(raw_path)==r['ocr_evidence']['sha256']
        assert raw['runtime']['preprocessing']
        assert raw['selected_rotation_deg_clockwise']==r['ocr_evidence']['selected_rotation_deg_clockwise']
        assert not r['title_complete'] and not any(r.get(k) for k in ('title_text','author_text','edition_text','isbn_text'))
        f=frames[r['frame_id']];fp=RUN/f['image_path'];assert digest(fp)==r['provenance']['frame_sha256']
        if r['frame_id'] not in cache:cache[r['frame_id']]=cv2.imread(str(fp))
        o=copy.deepcopy(obs[r['observation_id']])
        if r['crop_transform'].get('ordered_source_quad_xy'):o['rectify_quad_xy']=r['crop_transform']['ordered_source_quad_xy']
        expected,_=native_crop(cache[r['frame_id']],o)
        assert np.array_equal(expected,cv2.imread(str(crop)))


def test_control_reader_benchmark_saved_numbers_recomputed_using_original_evaluator():
    scripts=ROOT/'runs/worker6_readers_v2/scripts'
    if not (scripts/'dev_eval.py').exists():pytest.skip('reader benchmark artifacts absent')
    sys.path.insert(0,str(scripts))
    try:
        import dev_eval as D
        labels={l['observation_id']:l for l in D.LABELS['labels']}
        for engine,wanted,hits in [('rapid_v5_mobile_W1wrapper(2stage)',.2630081300813008,11),('florence2_base_letterbox_b1_90_270',.38008130081300817,13)]:
            d=json.loads((scripts.parent/'dev'/('dev_'+engine+'.json')).read_text())
            rc=[D.recall(labels[o['observation_id']],o['text']) for o in d['per_book']]
            assert len(rc)==41 and sum(rc)/len(rc)==pytest.approx(wanted)
            assert sum(v>=.5 for v in rc)==hits
    finally:sys.path.remove(str(scripts))
