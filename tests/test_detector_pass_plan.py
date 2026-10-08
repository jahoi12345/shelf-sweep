"""Sampling preserves source provenance; baseline remains a full per-frame pass."""
import pytest
from providers.local_detector import frame_pass_plan


@pytest.mark.parametrize('profile,size', [('fast_960',960),('fast_1024',1024),('fast_1280_512',1280)])
def test_sparse_item_schedule_is_explicit(profile,size):
    plans = [frame_pass_plan(profile, i) for i in range(6)]
    assert [p['run_items'] for p in plans] == [True,False,False,True,False,False]
    assert all(p['image_size'] == size and p['item_stride'] == 3 for p in plans)


def test_baseline_never_skips_nonbook_pass():
    assert all(frame_pass_plan('baseline', i) == {'image_size':1280,'item_stride':1,'run_items':True} for i in range(9))
    assert frame_pass_plan('baseline',0,1024)['image_size'] == 1024


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError):
        frame_pass_plan('unknown',0)


def test_five_frame_item_sampling_keeps_native_spine_resolution():
    plans=[frame_pass_plan('fast_1280_512_n5',i) for i in range(10)]
    assert [p['run_items'] for p in plans] == [True,False,False,False,False,True,False,False,False,False]
    assert all(p['item_stride']==5 and p['image_size']==1280 for p in plans)


def test_fast_items_schedule_and_prompt_families():
    from providers.local_detector import FAST_ITEMS_PROMPTS, NON_BOOK_PROMPTS, ITEM_REVIEW_FAMILIES, DESK_OFFICE_PROMPTS
    plans = [frame_pass_plan('fast_items', i) for i in range(6)]
    assert [p['run_items'] for p in plans] == [True] * 6
    assert all(p['image_size'] == 1280 and p['item_stride'] == 1 for p in plans)
    assert FAST_ITEMS_PROMPTS[:12] == NON_BOOK_PROMPTS
    assert len(FAST_ITEMS_PROMPTS) == 26 + len(DESK_OFFICE_PROMPTS) == 37
    assert FAST_ITEMS_PROMPTS[26:] == DESK_OFFICE_PROMPTS
    assert set(ITEM_REVIEW_FAMILIES[c] for _, c in FAST_ITEMS_PROMPTS) == {
        'potted', 'framed', 'container', 'decor', 'furniture/structure', 'electronics'}
    assert ITEM_REVIEW_FAMILIES['cup'] == 'container'
    assert [c for p, c in FAST_ITEMS_PROMPTS[22:26]] == ['plant'] * 4
    assert ITEM_REVIEW_FAMILIES['ornament'] == 'decor'


def test_process_default_is_fast_items_and_baseline_stays_selectable(monkeypatch,tmp_path):
    from providers.local_detector_process import LocalDetectorProcess
    monkeypatch.delenv('LIBRARY_LOCAL_DETECTOR_PROFILE',raising=False)
    assert LocalDetectorProcess(work_dir=tmp_path).profile == 'fast_items'
    monkeypatch.setenv('LIBRARY_LOCAL_DETECTOR_PROFILE','baseline')
    assert LocalDetectorProcess(work_dir=tmp_path).profile == 'baseline'


def test_process_profile_override_preserves_explicit_baseline(monkeypatch,tmp_path):
    from providers.local_detector_process import LocalDetectorProcess
    monkeypatch.setenv('LIBRARY_LOCAL_DETECTOR_PROFILE','fast_1024')
    assert LocalDetectorProcess(work_dir=tmp_path).profile == 'fast_1024'
    assert LocalDetectorProcess(work_dir=tmp_path,profile='baseline').profile == 'baseline'
    monkeypatch.setenv('LIBRARY_LOCAL_DETECTOR_PROFILE','invalid')
    with pytest.raises(ValueError):LocalDetectorProcess(work_dir=tmp_path)
