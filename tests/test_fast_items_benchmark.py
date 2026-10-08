"""Relative development metrics must not disguise class conflicts as coverage."""
from scripts.benchmark_fast_items import project_box, visual_coverage


def test_family_conflict_stays_separate_from_geometric_hit():
    visual={'label':'DEV/model-read, not human truth','objects':[
        {'id':'V01','name':'ornament','class':'decor','certainty':'clear','box':[0,0,10,10]},
        {'id':'V02','name':'jar','class':'vase_pot','certainty':'partial','box':[20,0,30,10]}]}
    observations=[
        {'observation_id':'wrong','kind':'non_book','category':'cup','frame_id':'f0','bbox_xyxy':[0,0,10,10]},
        {'observation_id':'right','kind':'non_book','category':'jar','frame_id':'f0','bbox_xyxy':[20,0,30,10]},
        {'observation_id':'book','kind':'book','category':'book','frame_id':'f0','bbox_xyxy':[0,0,10,10]},
        {'observation_id':'unstitched','kind':'non_book','category':'ornament','frame_id':'f1','bbox_xyxy':[0,0,10,10]}]
    result=visual_coverage(observations,visual,{'f0':[[1,0,0],[0,1,0],[0,0,1]]})
    assert result['hit_any_category']==2
    assert result['hit_compatible_family']==1
    assert result['by_class']['decor']['hit_compatible_family']==0
    assert [m['observation_id'] for m in result['objects'][0]['matches']]==['wrong']


def test_projection_uses_native_homography_and_homogeneous_division():
    assert project_box([0,0,10,20],[[2,0,6],[0,4,8],[0,0,2]])==[3,4,13,44]
