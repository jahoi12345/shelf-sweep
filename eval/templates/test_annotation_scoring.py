"""Synthetic invariant checks. These are software fixtures, NEVER human ground truth."""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import httpx
import numpy as np

from eval.score import score, match_boxes
from pipeline.read_spines import native_crop, sanitize_reading, read
from pipeline.resolve import resolve_one, isbn_value
from providers.openlibrary import OpenLibrary


def reading(title='The Example Atlas', author='Jane Doe', visible=None, **extra):
    return dict(observation_id='f00000-o000', visible_text=visible or f'{title}\n{author}',
                title_text=title, title_complete=True, author_text=author, publisher_text=None,
                read_confidence=.95, issues=[], **extra)


class Catalogue:
    requests=0
    def __init__(self, docs=None, count=None, edition=None):
        self.docs = docs if docs is not None else [dict(key='/works/OL1W',title='The Example Atlas',author_name=['Jane Doe'])]
        self.count = len(self.docs) if count is None else count
        self.edition = edition
        self.calls=[]
    def search(self,title,author):
        self.calls.append((title,author))
        return {'payload':{'docs':self.docs,'numFound':self.count},'url':'https://openlibrary.org/search.json?fixture','retrieved_at':'synthetic'}
    def edition_by_isbn(self,isbn):
        return {'payload':self.edition,'url':f'https://openlibrary.org/isbn/{isbn}.json','retrieved_at':'synthetic'}


def labelled_fixture():
    # Values describe synthetic rectangles, not books in the captured room.
    objects=[dict(annotation_id='h1',instance_id='human-book-a',bbox_xyxy=[0,0,20,80],kind='book',category='book',title_legible=True,accepted_titles=['Alpha Atlas']),
             dict(annotation_id='h2',instance_id='human-book-b',bbox_xyxy=[30,0,50,80],kind='book',category='book',title_legible=True,accepted_titles=['Beta Atlas']),
             dict(annotation_id='h3',instance_id='human-book-c',bbox_xyxy=[60,0,80,80],kind='book',category='book',title_legible=True,accepted_titles=['Gamma Atlas']),
             dict(annotation_id='h4',instance_id='human-item-a',bbox_xyxy=[90,0,120,80],kind='non_book',category='plant',title_legible=None,accepted_titles=[])]
    annotation={'annotation_version':1,'provenance':{'source_type':'human','annotator':'synthetic-test-declaration-A','reviewer':'synthetic-test-declaration-B','review_status':'approved','frozen_at':'synthetic'},
                'coverage':{'whole_video':False,'physical_instances_complete':True,'physical_inventory_complete':False},
                'frames':[{'frame_id':'f00000','width_px':200,'height_px':100,'exhaustive':True,'reviewed':True,'objects':objects}]}
    detections=[dict(observation_id='f00000-o000',frame_id='f00000',bbox_xyxy=[0,0,20,80],kind='book'),
                dict(observation_id='f00000-o001',frame_id='f00000',bbox_xyxy=[30,0,50,80],kind='book'),
                dict(observation_id='f00000-o002',frame_id='f00000',bbox_xyxy=[90,0,120,80],kind='book')]
    readings={'data':{'readings':[dict(observation_id='f00000-o000',title_text='Alpha Atlas',read_confidence=.9),dict(observation_id='f00000-o001',title_text='Wrong Atlas',read_confidence=.95)]}}
    candidates={'data':{'observations':detections}}
    return annotation,candidates,readings


class CropChecks(unittest.TestCase):
    def test_rotation_inverse_maps_pixels_to_frame(self):
        image=np.arange(12*14,dtype=np.uint8).reshape(12,14)
        for rotation in (0,90,180,270):
            crop, transform=native_crop(image,{'bbox_xyxy':[3,2,10,11],'text_rotation_deg':rotation})
            matrix=np.asarray(transform['crop_to_frame_3x3'])
            for y in range(crop.shape[0]):
                for x in range(crop.shape[1]):
                    point=matrix@np.array([x,y,1])
                    self.assertEqual(crop[y,x],image[int(point[1]),int(point[0])])
            self.assertFalse(transform['upscaled'])
    def test_invalid_geometry_refused(self):
        for box in ([-1,0,4,5],[2,2,2,8],[0,0,20,40],[0,0,float('nan'),4]):
            with self.assertRaises(ValueError):native_crop(np.zeros((10,10,3),np.uint8),{'bbox_xyxy':box})
    def test_rectification_transform(self):
        _,t=native_crop(np.zeros((50,50,3),np.uint8),{'bbox_xyxy':[5,5,40,40],'rectify_quad_xy':[[10,10],[30,10],[30,35],[10,35]]})
        matrix=np.array(t['crop_to_frame_3x3']);p=matrix@np.array([0,0,1])
        self.assertTrue(np.allclose(p[:2]/p[2],[10,10]))


class ReadingChecks(unittest.TestCase):
    def test_inferred_and_incomplete_identity_abstains(self):
        out=sanitize_reading({**reading(visible='ATLAS [?]'),'title_complete':False})
        self.assertIsNone(out['title_text']);self.assertIsNone(out['author_text'])
    def test_unknown_not_promoted_by_confidence(self):
        out=sanitize_reading({**reading(),'visible_text':None})
        self.assertIsNone(out['title_text']);self.assertFalse(out['title_complete'])
    def test_failed_provider_attempts_respect_budget(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cv2.imwrite(str(root/'f.png'),np.zeros((100,100,3),np.uint8))
            manifest={'run_id':'synthetic','data':{'frames':[{'frame_id':'f00000','image_path':'f.png','timestamp_s':0,'width_px':100,'height_px':100}]}}
            candidates={'run_id':'synthetic','data':{'observations':[dict(observation_id=f'f00000-o{i:03}',frame_id='f00000',kind='book',bbox_xyxy=[i,0,i+10,50]) for i in range(5)]}}
            for name,value in [('m.json',manifest),('c.json',candidates)]: (root/name).write_text(json.dumps(value))
            calls=[]
            async def failing(**kwargs):calls.append(kwargs);raise RuntimeError('synthetic failure')
            out=json.loads(read(root/'c.json',root/'m.json',root,{'image_request':failing,'max_books':2}).read_text())
            self.assertEqual(len(calls),2);self.assertEqual(out['metrics']['crops_saved'],5)
            self.assertEqual(out['data']['readings'][2]['issues'],['reading_budget_limit'])


class SharedProviderChecks(unittest.TestCase):
    def test_shared_adapter_native_bytes_cache_and_separate_resolution(self):
        from providers.gemini import set_transport, make_adapter, UsageLog, get_cache
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            image=np.zeros((80,60,3),np.uint8)
            cv2.putText(image,'ATLAS',(2,30),cv2.FONT_HERSHEY_SIMPLEX,.3,(255,255,255),1)
            cv2.imwrite(str(root/'frame.png'),image)
            manifest={'run_id':'synthetic_fixture','data':{'frames':[dict(frame_id='f00000',image_path='frame.png',timestamp_s=0,width_px=60,height_px=80)]}}
            candidate={'run_id':'synthetic_fixture','fixture':True,'data':{'observations':[dict(observation_id='f00000-o000',frame_id='f00000',kind='book',bbox_xyxy=[0,0,60,80])]}}
            for name,value in [('manifest.json',manifest),('candidates.json',candidate)]:
                (root/name).write_text(json.dumps(value))
            calls=[]
            async def transport(model,images,prompt,config,schema):
                calls.append(images)
                return json.dumps(reading()),{'prompt_tokens':10,'output_tokens':10,'thought_tokens':0,'total_tokens':20}
            usage=UsageLog()
            adapter=make_adapter(run_dir=root,cache=get_cache(root/'cache'),usage=usage,model='synthetic-provider')
            set_transport(transport)
            try:
                first=json.loads(read(root/'candidates.json',root/'manifest.json',root,{'image_request':adapter}).read_text())
                second=json.loads(read(root/'candidates.json',root/'manifest.json',root,{'image_request':adapter}).read_text())
                self.assertEqual(len(calls),1)
                self.assertEqual(first['data']['readings'][0]['visible_text'],'The Example Atlas\nJane Doe')
                self.assertNotIn('work',first['data']['readings'][0])
                self.assertEqual(second['metrics']['provider_metrics'][0]['cache_hits'],1)
                self.assertEqual(calls[0][0][0],(root/first['data']['readings'][0]['crop_path']).read_bytes())
                self.assertEqual(usage.summary('read')['requests'],1)
            finally:
                set_transport(None)


class ResolutionChecks(unittest.TestCase):
    def test_unreadable_never_queries(self):
        cat=Catalogue();out=resolve_one({**reading(),'visible_text':None},cat)
        self.assertEqual(out['decision'],'no_match');self.assertEqual(cat.calls,[])
    def test_catalogue_does_not_supply_unseen_author_or_edition(self):
        out=resolve_one(reading(author=None,visible='The Example Atlas'),Catalogue())
        self.assertEqual(out['decision'],'accepted');self.assertIsNone(out['author']);self.assertIsNone(out['edition']);self.assertIsNone(out['isbn'])
    def test_generic_title_without_author_abstains(self):
        out=resolve_one(reading(title='Politics',author=None,visible='Politics'),Catalogue([dict(key='/works/OL1W',title='Politics',author_name=['Famous Author'])]))
        self.assertEqual(out['decision'],'ambiguous');self.assertIsNone(out['title'])
    def test_conflicting_works_and_truncated_search_abstain(self):
        for cat in (Catalogue(count=50),Catalogue([dict(key='/works/OL1W',title='The Example Atlas',author_name=['Jane Doe']),dict(key='/works/OL2W',title='The Example Atlas',author_name=['Jane Doe'])])):
            self.assertEqual(resolve_one(reading(),cat)['decision'],'ambiguous')
    def test_edition_requires_printed_checksum_valid_isbn_work_title(self):
        isbn='9780140328721'
        ed={'key':'/books/OL2M','title':'The Example Atlas','works':[{'key':'/works/OL1W'}],'isbn_13':[isbn]}
        r=reading(visible=f'The Example Atlas\nJane Doe\n{isbn}',isbn_text=isbn)
        self.assertEqual(resolve_one(r,Catalogue(edition=ed))['isbn'],isbn)
        ed['works']=[{'key':'/works/OL99W'}]
        self.assertIsNone(resolve_one(r,Catalogue(edition=ed))['isbn'])
        self.assertIsNone(isbn_value('9780140328722'))
    def test_cache_retrieval_retains_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            transport=httpx.MockTransport(lambda req:httpx.Response(200,json={'numFound':0,'docs':[]}))
            cat=OpenLibrary(d,online=True,transport=transport)
            first=cat.search('Synthetic Example');cat.close()
            cat=OpenLibrary(d,online=False);second=cat.search('Synthetic Example')
            self.assertEqual(first,second);self.assertEqual(cat.requests,0);cat.close()


class ScoringChecks(unittest.TestCase):
    def test_human_legible_denominator_includes_missed_books_and_clutter(self):
        a,c,r=labelled_fixture();out=score(a,c,readings=r)
        m=out['metrics']
        self.assertEqual((m['book_detection_precision']['numerator'],m['book_detection_precision']['denominator']),(2,3))
        self.assertEqual((m['literal_title_correctness_observations']['numerator'],m['literal_title_correctness_observations']['denominator']),(1,3))
        self.assertEqual(out['title_observations']['literal']['missed'],1)
        self.assertEqual(m['confident_wrong_observations']['numerator'],1)
        self.assertEqual(out['clutter_false_positives']['counts_by_category'],{'plant':1})
        self.assertFalse(m['book_count_error']['available'])
    def test_model_labels_and_pending_review_are_unavailable(self):
        a,c,r=labelled_fixture();a['provenance']['source_type']='model'
        self.assertEqual(score(a,c,readings=r)['status'],'unavailable')
        a['provenance']['source_type']='human';a['provenance']['review_status']='pending'
        self.assertEqual(score(a,c)['status'],'unavailable')
    def test_split_merge_and_duplicate_denominators(self):
        a,c,r=labelled_fixture()
        inventory={'data':{'books':[dict(instance_id='bk-000',observation_ids=['f00000-o000','f00000-o001'],title='Alpha Atlas',confidence=.9),dict(instance_id='bk-001',observation_ids=['f00000-o000'],title='Alpha Atlas',confidence=.9)],'items':[]}}
        out=score(a,c,readings=r,inventory=inventory);m=out['metrics']
        self.assertEqual((m['merged_instances']['numerator'],m['merged_instances']['denominator']),(1,2))
        self.assertEqual((m['split_instances']['numerator'],m['split_instances']['denominator']),(1,4))
        self.assertEqual((m['duplicate_instances']['numerator'],m['duplicate_instances']['denominator']),(1,2))
        self.assertEqual(m['title_correctness']['denominator'],3)
        self.assertIsNone(m['title_correctness']['pass'])
    def test_nested_identity_is_scored(self):
        a,c,r=labelled_fixture()
        inventory={'data':{'books':[dict(instance_id='bk-000',observation_ids=['f00000-o000'],identity={'title':'Alpha Atlas','id_confidence':.9})],'items':[]}}
        out=score(a,c,inventory=inventory)
        self.assertEqual(out['metrics']['title_correctness']['numerator'],1)
        self.assertEqual(out['title_instances']['missed'],2)

    def test_missing_non_book_category_does_not_count_correct(self):
        a,c,r=labelled_fixture();c['data']['observations'][2]['kind']='non_book'
        out=score(a,c,readings=r)
        self.assertEqual(out['metrics']['non_book_detection_coverage']['numerator'],1)
        self.assertEqual(out['metrics']['non_book_correct_category_coverage_observations']['numerator'],0)
    def test_matching_maximizes_cardinality_before_iou(self):
        truth=[{'bbox_xyxy':[0,0,10,10]},{'bbox_xyxy':[1,0,11,10]}]
        pred=[{'bbox_xyxy':[0,0,10,10]},{'bbox_xyxy':[-2,0,8,10]}]
        out=match_boxes(truth,pred,.6)
        self.assertEqual(len(out),2)
        self.assertEqual({(x,y) for x,y,_ in out},{(0,1),(1,0)})
    def test_unprocessed_frame_misses_count(self):
        a,c,r=labelled_fixture();a['frames'].append({**a['frames'][0],'frame_id':'f00001','objects':[{**a['frames'][0]['objects'][0],'annotation_id':'h5','instance_id':'human-book-d'}]})
        out=score(a,c,readings=r)
        self.assertEqual(out['metrics']['book_detection_recall']['denominator'],4)
        self.assertEqual(out['title_observations']['literal']['missed'],2)


if __name__ == '__main__':unittest.main()
