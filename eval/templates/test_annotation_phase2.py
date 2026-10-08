"""Synthetic software checks for review attribution, bilingual evidence and comparisons.

Not real books, human labels, real-provider behavior or bilingual accuracy.
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from eval.score import annotation_problems, score
from eval.compare import compare
from eval.templates.test_annotation_scoring import labelled_fixture, reading, Catalogue
from pipeline.read_spines import sanitize_reading, READ_SCHEMA, ordered_quad, native_crop
import numpy as np
from eval.templates.us_pricing_probe import parse_offer
from pipeline.resolve import resolve_one, normalize_title


class BilingualChecks(unittest.TestCase):
    def test_accents_preserved_and_untranslated_fields_rejected(self):
        literal='Cien años de soledad\nGabriel García Márquez'
        r=sanitize_reading({**reading(title='Cien años de soledad',author='Gabriel García Márquez',visible=literal),
                            'language_code':'es','language_evidence':'años de soledad'})
        self.assertEqual(r['visible_text'],literal);self.assertEqual(r['language_code'],'es')
        wrong=sanitize_reading({**r,'title_text':'One Hundred Years of Solitude'})
        self.assertIsNone(wrong['title_text']);self.assertEqual(wrong['visible_text'],literal)
        self.assertNotEqual(normalize_title('El sí de las niñas'),normalize_title('El si de las niñas'))
    def test_language_requires_literal_evidence(self):
        r=sanitize_reading({**reading(),'language_code':'es','language_evidence':'Spanish edition'})
        self.assertIsNone(r['language_code'])
    def test_spanish_native_title_can_support_work_via_spanish_catalogue_edition(self):
        class SpanishCatalogue(Catalogue):
            def search(self,title,author,language=None):
                self.requested_language=language
                return super().search(title,author)
        cat=SpanishCatalogue([{'key':'/works/OL1W','title':'One Hundred Years of Solitude','author_name':['Gabriel García Márquez'],
                               'editions':{'docs':[{'key':'/books/OL2M','title':'Cien años de soledad','language':['spa']}]}}])
        r={**reading(title='Cien años de soledad',author='García Márquez'), 'language_code':'es','language_evidence':'años de soledad'}
        out=resolve_one(r,cat)
        self.assertEqual(cat.requested_language,'es');self.assertEqual(out['decision'],'accepted')
        self.assertEqual(out['title'],'Cien años de soledad')
        self.assertEqual(out['catalogue_title'],'One Hundred Years of Solitude')
        self.assertIsNone(out['edition']);self.assertIsNone(out['isbn'])
    def test_translated_only_catalogue_hit_does_not_replace_native_ocr(self):
        cat=Catalogue([{'key':'/works/OL1W','title':'One Hundred Years of Solitude','author_name':['Gabriel García Márquez']}])
        out=resolve_one(reading(title='Cien años de soledad',author='García Márquez'),cat)
        self.assertIsNone(out['title']);self.assertEqual(out['title_text'],'Cien años de soledad')
    def test_printed_isbn_cannot_select_contradictory_language_edition(self):
        isbn='9780140328721'
        r=reading(visible='The Example Atlas\nJane Doe\n'+isbn,isbn_text=isbn)
        r.update(language_code='en',language_evidence='The Example Atlas')
        class LangCatalogue(Catalogue):
            def search(self,title,author,language=None):return super().search(title,author)
        cat=LangCatalogue(edition={'key':'/books/OL2M','title':'The Example Atlas','isbn_13':[isbn],
                                  'works':[{'key':'/works/OL1W'}],'languages':[{'key':'/languages/spa'}]})
        out=resolve_one(r,cat);self.assertEqual(out['decision'],'accepted');self.assertIsNone(out['edition'])
    def test_human_language_groups_include_misses_and_wrong_confident_titles(self):
        a,c,r=labelled_fixture()
        a['frames'][0]['objects'][0]['language_code']='es'
        a['frames'][0]['objects'][1]['language_code']='en'
        a['frames'][0]['objects'][2]['language_code']='es'
        # Predicted language deliberately wrong; grouping still uses human labels.
        r['data']['readings'][0]['language_code']='en'
        out=score(a,c,readings=r)['language_scores']['literal_observations']
        self.assertEqual(out['es']['human_legible_sample_size'],2)
        self.assertEqual(out['es']['correctness']['value'],.5)
        self.assertEqual(out['es']['counts']['missed'],1)
        self.assertEqual(out['en']['human_legible_sample_size'],1)
        self.assertEqual(out['en']['counts']['confident_wrong'],1)
        self.assertFalse(out['mul']['correctness']['available'])
    def test_nullable_bilingual_schema_is_sdk_compatible(self):
        try:from google.genai import _transformers
        except ImportError:self.skipTest('SDK unavailable in this Python runtime')
        schema=_transformers.t_schema(None,READ_SCHEMA)
        self.assertEqual(len(schema.properties),11)
        self.assertTrue(schema.properties['read_confidence'].nullable)


class GeometryAndPriceChecks(unittest.TestCase):
    def test_unordered_quad_rectification_maps_corners_back_to_native_evidence(self):
        q=ordered_quad([[38,42],[32,8],[12,11],[16,45]])
        crop,t=native_crop(np.zeros((60,60,3),np.uint8),{'bbox_xyxy':[10,5,45,50],'rectify_quad_xy':q})
        matrix=np.array(t['crop_to_frame_3x3'])
        corners=[[0,0,1],[crop.shape[1]-1,0,1],[crop.shape[1]-1,crop.shape[0]-1,1],[0,crop.shape[0]-1,1]]
        for point,expected in zip(corners,q):
            p=matrix@point;self.assertTrue(np.allclose(p[:2]/p[2],expected,atol=.001))
        self.assertFalse(t['upscaled'])
    def test_geometry_ties_abstain_and_do_not_establish_text_rotation(self):
        with self.assertRaises(ValueError):ordered_quad([[20,0],[40,20],[20,40],[0,20]])
    def test_price_requires_exact_isbn_usd_us_and_physical_format(self):
        book={'@type':'Book','isbn':'9780142410370','bookFormat':'https://schema.org/Paperback',
              'potentialAction':{'expectsAcceptanceOf':{'@type':'Offer','Price':9.99,'priceCurrency':'USD','eligibleRegion':{'name':'US'}}}}
        def html(b):return '<script type="application/ld+json">'+json.dumps(b)+'</script>'
        self.assertEqual(parse_offer(html(book),'9780142410370')['amount'],9.99)
        for path,value in [('bookFormat','https://schema.org/EBook'),('isbn','9780140328721')]:
            b=copy.deepcopy(book);b[path]=value
            with self.assertRaises(ValueError):parse_offer(html(b),'9780142410370')
        b=copy.deepcopy(book);b['potentialAction']['expectsAcceptanceOf']['priceCurrency']='GBP'
        with self.assertRaises(ValueError):parse_offer(html(b),'9780142410370')


class ReviewChecks(unittest.TestCase):
    def test_one_human_self_review_and_frozen_labels_are_sufficient(self):
        a,c,r=labelled_fixture()
        a['provenance'].update(annotator='Named human',reviewer='Named human',review_mode='self_reviewed')
        self.assertEqual(annotation_problems(a),[])
        self.assertEqual(score(a,c,readings=r)['status'],'development_subset')
    def test_model_unreviewed_unfrozen_and_false_independent_review_rejected(self):
        a,_,_=labelled_fixture()
        a['provenance'].update(annotator='Named human',reviewer='Named human',review_mode='independent')
        self.assertTrue(annotation_problems(a))
        a['provenance']['review_mode']='self_reviewed'
        for field,value in [('source_type','model'),('review_status','pending'),('frozen_at',None)]:
            b=copy.deepcopy(a);b['provenance'][field]=value;self.assertTrue(annotation_problems(b))
    def test_score_does_not_mutate_frozen_annotations(self):
        a,c,r=labelled_fixture();before=json.dumps(a,sort_keys=True)
        score(a,c,readings=r);self.assertEqual(before,json.dumps(a,sort_keys=True))
    def test_missing_truth_never_reports_zero_clutter_errors(self):
        a,c,r=labelled_fixture();a['provenance']['review_status']='pending'
        out=score(a,c,readings=r)
        self.assertIsNone(out['clutter_false_positives']['counts_by_category'])
        self.assertIsNone(out['metrics']['clutter_false_positive_rate']['denominator'])
        self.assertFalse(out['metrics']['duplicate_predictions']['available'])
    def test_blocked_comparison_has_unavailable_accuracy_not_empty_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a,_,_=labelled_fixture()
            (root/'labels.json').write_text(json.dumps(a))
            (root/'entries.json').write_text(json.dumps([{'name':'reader','run_dir':str(root),'execution_status':'blocked','reason':'weights unavailable'}]))
            path=compare(root/'labels.json',root/'entries.json',root/'output')
            out=json.loads(path.read_text())['runs'][0]
            self.assertEqual(out['execution_status'],'blocked')
            self.assertIsNone(out['evaluation']['language_scores']['literal_observations']['es']['human_legible_sample_size'])
            self.assertFalse(out['evaluation']['metrics']['title_correctness']['available'])
            with self.assertRaises(ValueError):compare(root/'labels.json',root/'entries.json',root/'output')

if __name__=='__main__':unittest.main()
