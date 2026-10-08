"""Setup failures must preserve resumable bytes without publishing bad weights."""
import hashlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import pytest

spec=importlib.util.spec_from_file_location('setup_models',Path(__file__).resolve().parents[1]/'scripts/setup_local_models.py')
setup=importlib.util.module_from_spec(spec);spec.loader.exec_module(setup)

class Response(io.BytesIO):
    def __init__(self,data,status=200,headers=None):super().__init__(data);self.status=status;self.headers=headers or {}


def test_interrupted_download_resumes_and_checks_hash(tmp_path):
    data=b'synthetic model bytes';dest=tmp_path/'weight';part=tmp_path/'weight.part';part.write_bytes(data[:9]);calls=[]
    def open_(request,**kwargs):
        calls.append(request.get_header('Range'))
        return Response(data[9:],206,{'Content-Range':f'bytes 9-{len(data)-1}/{len(data)}'})
    setup.fetch('https://fixture.invalid/model',dest,hashlib.sha256(data).hexdigest(),opener=open_)
    assert calls==['bytes=9-'] and dest.read_bytes()==data and not part.exists()
    setup.fetch('https://fixture.invalid/model',dest,hashlib.sha256(data).hexdigest(),opener=lambda *a,**kw:pytest.fail('cached checked file must not download'))


def test_server_ignoring_range_restarts_instead_of_appending(tmp_path):
    data=b'complete synthetic model';dest=tmp_path/'weight';(tmp_path/'weight.part').write_bytes(b'obsolete prefix')
    setup.fetch('https://fixture.invalid/model',dest,hashlib.sha256(data).hexdigest(),opener=lambda *a,**kw:Response(data))
    assert dest.read_bytes()==data


def test_checksum_mismatch_never_publishes_destination(tmp_path):
    dest=tmp_path/'weight'
    with pytest.raises(RuntimeError):setup.fetch('https://fixture.invalid/model',dest,'0'*64,opener=lambda *a,**kw:Response(b'wrong'))
    assert not dest.exists() and not (tmp_path/'weight.part').exists()


def test_failed_network_keeps_partial_file(tmp_path):
    part=tmp_path/'weight.part';part.write_bytes(b'previous valid prefix')
    def failing(*a,**kw):raise ConnectionResetError('synthetic outage')
    with pytest.raises(RuntimeError):setup.fetch('https://fixture.invalid/model',tmp_path/'weight','0'*64,opener=failing)
    assert part.read_bytes()==b'previous valid prefix'


def test_settings_switch_modes_preserving_unrelated_configuration(tmp_path):
    env=tmp_path/'.env';env.write_text('# settings\nUNRELATED_TOKEN=synthetic-placeholder\nLIBRARY_DETECTOR=local\nLIBRARY_DETECTOR=auto\n')
    setup.settings(env,{'LIBRARY_DETECTOR':'none','LIBRARY_READER':'rapidocr'})
    assert env.read_text().count('LIBRARY_DETECTOR=')==1
    assert 'UNRELATED_TOKEN=synthetic-placeholder' in env.read_text() and 'LIBRARY_READER=rapidocr' in env.read_text()
    setup.settings(env,{'LIBRARY_DETECTOR':'local','LIBRARY_READER':'rapidocr'})
    assert 'LIBRARY_DETECTOR=local' in env.read_text()


def test_timed_out_installer_is_stopped(tmp_path,monkeypatch):
    monkeypatch.setattr(setup,'ROOT',tmp_path)
    with pytest.raises(subprocess.TimeoutExpired):setup.command([sys.executable,'-c','import time;time.sleep(10)'],timeout=.1)


def test_minimal_excludes_large_runtimes(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup,'ROOT',tmp_path)
    commands=[];monkeypatch.setattr(setup,'command',lambda args,**kw:commands.append([str(x) for x in args]))
    monkeypatch.setattr(setup.subprocess,'check_output',lambda *a,**kw:str(tmp_path/'rapid_models'))
    monkeypatch.setattr(setup,'fetch',lambda *a,**kw:None)
    assert setup.main(['--minimal'])==0
    flat=' '.join(' '.join(x) for x in commands)
    assert all(name not in flat for name in ['torch==','ultralytics==','git+','florence','easyocr=='])
    assert 'LIBRARY_DETECTOR=none' in (tmp_path/'.env').read_text()
    timings=list((tmp_path/'.runtime/setup_timings').glob('*.json'))
    assert len(timings)==1 and '"status": "ok"' in timings[0].read_text()
