"""Verify whole-video provenance and summarize usage; never infer ground truth."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from pipeline import contracts as C
from pipeline.detect import read_json,save_json


def validate(source,baseline,improvement,labels,out):
    source,baseline,improvement=map(Path,[source,baseline,improvement])
    src=read_json(source/'manifest.json');checks=[];usage=[]
    for run in [baseline,improvement]:
        manifest=read_json(run/'manifest.json');candidates=read_json(run/'candidates.json');inventory=read_json(run/'inventory.json')
        if manifest['data']['frames']!=src['data']['frames']:
            raise ValueError('Whole-video IDs/timestamps/selection changed')
        if not manifest['run_id']==candidates['run_id']==inventory['run_id']==run.name:
            raise ValueError('Mixed run IDs')
        obs={o['observation_id']:o for o in candidates['data']['observations']}
        errors=C.validate_manifest(manifest,run)+C.validate_candidates(candidates,manifest)+C.validate_inventory(inventory,manifest,set(obs))
        if errors:raise ValueError('; '.join(errors[:5]))
        for f in manifest['data']['frames']:
            if hashlib.sha256((run/f['image_path']).read_bytes()).digest()!=hashlib.sha256((source/f['image_path']).read_bytes()).digest():
                raise ValueError('Evidence bytes changed')
        for item in inventory['data']['books']+inventory['data']['items']:
            ids=[obs[x]['frame_id'] for x in item['observation_ids']]
            if len(ids)!=len(set(ids)):raise ValueError('Same-frame physical merge')
        checks.append({'run':run.name,'native_evidence_hashes_verified':len(manifest['data']['frames']),
            'manifest_candidates_inventory_valid':True,'no_same_frame_instance_merge':True})
        # Reused versions deliberately omit original request rows; their source
        # rows are nested as historical provenance and must not be rebilled.
        requests=candidates['metrics'].get('requests',[])
        settings=read_json(run/'local_provider_settings.json') if (run/'local_provider_settings.json').exists() else {}
        usage.append({'run':run.name,'fresh_local_inferences':sum(r.get('local_inference_count',0) for r in requests),
            'actual_API_requests':sum(r.get('request_count',0) for r in requests),'local_API_cost':0,
            'detection_elapsed_s':candidates['metrics']['elapsed_s'],'local_inference_sum_s':sum(r.get('latency_s',0) for r in requests),
            'setup_s':settings.get('setup_s'),'dedup_elapsed_s':inventory['metrics']['elapsed_s'],
            'reuse':'Identical baseline detections; no new inference' if run==improvement else 'Fresh local CPU inference; cached model weights, no prediction cache',
            'counts':inventory['data']['counts'],'trusted_registration_pairs':sum(r['status']=='accepted' for r in inventory['data']['registrations'])})
    if read_json(baseline/'candidates.json')['data']!=read_json(improvement/'candidates.json')['data']:
        raise ValueError('Comparison observations differ')
    output=Path(out)
    if output.exists():raise ValueError('Choose a new validation output version')
    return save_json(output,{'checks':checks,'baseline_improvement_candidate_data_identical':True,'actual_usage':usage,
        'labels_file_sha256':hashlib.sha256(Path(labels).read_bytes()).hexdigest(),
        'accuracy':'Unavailable until exhaustive independent human labels are attributable, reviewed and frozen; registration availability is not accuracy'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--baseline',required=True);p.add_argument('--improvement',required=True)
    p.add_argument('--labels',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(validate(a.source,a.baseline,a.improvement,a.labels,a.out))

if __name__=='__main__':main()
