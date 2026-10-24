"""CPU-only public Zarr API tests of the actual newly linked guarded library."""
import hashlib,json,os,pathlib,subprocess,sys,tempfile
def main():
 out=pathlib.Path(sys.argv[1]);exe=out/'integrated_reader_fixture';temporary=tempfile.TemporaryDirectory(prefix='integrated-fixtures-',dir=out);tests=pathlib.Path(temporary.name)
 cases=[('encoded','abc',False,False,'abc'),('storage','abc',False,False,'abc'),('encoded','abd',False,False,'SHA256 mismatch'),('encoded','abcd',False,False,'size mismatch'),('encoded',None,False,False,'missing/nonregular'),('encoded',None,True,False,'EXPLICIT_ABSENT'),('encoded','abc',True,False,'source-absent chunk appeared'),('encoded',None,False,True,'unlisted CT read'),('shard','abc',False,False,'bypass API forbidden'),('store','abc',False,False,'alternate store forbidden'),('store_registry','abc',False,False,'alternate store forbidden'),('level','abc',False,False,'unapproved CT level')]
 rows=[]
 metadata={'zarr_format':2,'shape':[1,1,3],'chunks':[1,1,3],'dtype':'|u1','compressor':None,'filters':None,'order':'C','fill_value':0,'dimension_separator':'/'}
 for number,(mode,data,absent,unlisted,expected) in enumerate(cases):
  root=tests/str(number);volume=root/'volume';group=volume/('1' if mode=='level' else '0');group.mkdir(parents=True)
  (volume/'0').mkdir(exist_ok=True);(group/'.zarray').write_text(json.dumps(metadata))
  if data is not None:
   chunk=group/'0/0/0';chunk.parent.mkdir(parents=True);chunk.write_bytes(data.encode())
  row={'key':'0/0/0','absent':absent,'bytes':0 if absent else 3,'sha256':'' if absent else hashlib.sha256(b'abc').hexdigest()}
  if unlisted:row['key']='9/9/9'
  manifest={'schema':'verified-ct-read-manifest-v1','volume_root':str(volume),'max_chunk_bytes':1024,'objects':[row]}
  path=root/'manifest.json';path.write_text(json.dumps(manifest));ledger=root/'events.jsonl'
  env=dict(os.environ,VC_VERIFIED_CT_MANIFEST=str(path),VC_VERIFIED_CT_MANIFEST_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),VC_VERIFIED_CT_LEDGER=str(ledger))
  store_root=root/'unrelated-store'
  store_array=store_root/'0';store_array.mkdir(parents=True)
  (store_array/'.zarray').write_text(json.dumps(metadata))
  store_chunk=store_array/'0/0/0';store_chunk.parent.mkdir(parents=True);store_chunk.write_bytes(b'abc')
  argument=store_root if mode in ('store','store_registry') else group
  r=subprocess.run([str(exe),str(argument),'encoded' if mode=='level' else mode],env=env,text=True,capture_output=True,timeout=15)
  success=expected in ('abc','EXPLICIT_ABSENT')
  assert r.returncode==(0 if success else 7),(number,r.returncode,r.stderr)
  assert expected in (r.stdout if success else r.stderr),(number,expected,r.stdout,r.stderr)
  events=[json.loads(line) for line in ledger.read_text().splitlines()]
  assert events[0]['status']=='INITIALIZED'
  if mode in ('store','store_registry'):
   assert any(e['status']=='REJECTED_STORE' and e['key']=='alternate-store' for e in events),(mode,events)
  rows.append({'case':number,'mode':mode,'expected':expected,'returncode':r.returncode,'events':events})
 # These subprocesses catch every std::exception in the public-API fixture.
 # Mandatory evidence failures must still terminate nonzero, not become fills.
 base_env=dict(env);base_env['VC_VERIFIED_CT_MANIFEST_SHA256']=hashlib.sha256(path.read_bytes()).hexdigest()
 # Use the first valid present fixture for runtime evidence tests.
 root=tests/'0';group=root/'volume/0';path=root/'manifest.json'
 base_env.update(VC_VERIFIED_CT_MANIFEST=str(path),VC_VERIFIED_CT_MANIFEST_SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
 for name,mode in [('init_write','encoded'),('read_write','ledgerfail'),('event_cap','eventlimit'),('bad_sha','encoded'),('incomplete','encoded')]:
  ledger=tests/('failure-'+name);case_env=dict(base_env,VC_VERIFIED_CT_LEDGER=str(ledger))
  if name=='init_write':ledger.mkdir()
  if name=='bad_sha':case_env['VC_VERIFIED_CT_MANIFEST_SHA256']='0'*64
  if name=='incomplete':case_env.pop('VC_VERIFIED_CT_LEDGER')
  r=subprocess.run([str(exe),str(group),mode],env=case_env,text=True,capture_output=True,timeout=30)
  assert r.returncode==86,(name,r.returncode,r.stdout,r.stderr)
  assert 'mandatory evidence failure' in r.stderr
  rows.append({'case':name,'returncode':r.returncode})
 # Simulate a renderer catching corrupt-read errors: rc0 is insufficient;
 # durable REJECTED evidence remains mandatory for the wrapper to reject it.
 corrupt=tests/'2';ledger=corrupt/'swallowed.jsonl';path=corrupt/'manifest.json'
 case_env=dict(base_env,VC_VERIFIED_CT_MANIFEST=str(path),VC_VERIFIED_CT_MANIFEST_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),VC_VERIFIED_CT_LEDGER=str(ledger))
 r=subprocess.run([str(exe),str(corrupt/'volume/0'),'swallow'],env=case_env,capture_output=True,timeout=15)
 assert r.returncode==0 and any(json.loads(x)['status']=='REJECTED' for x in ledger.read_text().splitlines())
 rows.append({'case':'swallowed_read_failure_has_REJECTED','returncode':r.returncode})
 # Every non-empty proper subset of the opt-in tuple is a fatal configuration error.
 all_keys=('VC_VERIFIED_CT_MANIFEST','VC_VERIFIED_CT_MANIFEST_SHA256','VC_VERIFIED_CT_LEDGER')
 values={'VC_VERIFIED_CT_MANIFEST':str(path),'VC_VERIFIED_CT_MANIFEST_SHA256':hashlib.sha256(path.read_bytes()).hexdigest()}
 for mask in (1,2,3,4,5,6):
  case_env=dict(base_env)
  for key in all_keys:case_env.pop(key,None)
  values['VC_VERIFIED_CT_LEDGER']=str(tests/('partial-env-'+str(mask)+'.jsonl'))
  for bit,key in enumerate(all_keys):
   if mask & (1<<bit):case_env[key]=values[key]
  r=subprocess.run([str(exe),str(group),'encoded'],env=case_env,capture_output=True,timeout=15)
  assert r.returncode==86 and b'mandatory evidence failure' in r.stderr,(mask,r.returncode,r.stdout,r.stderr)
  rows.append({'case':'partial_opt_in_mask_'+str(mask),'returncode':r.returncode})
 plain=dict(os.environ)
 for key in ('VC_VERIFIED_CT_MANIFEST','VC_VERIFIED_CT_MANIFEST_SHA256','VC_VERIFIED_CT_LEDGER'):plain.pop(key,None)
 r=subprocess.run([str(exe),str(group),'encoded'],env=plain,capture_output=True,timeout=15)
 assert r.returncode==0 and b'abc' in r.stdout
 rows.append({'case':'absent_opt_in_stock','returncode':r.returncode})
 # A valid digest over a malformed document must still fail closed at init.
 valid=json.loads(path.read_text())
 malformed_cases=[]
 malformed_cases.append(('malformed_json',b'{'))
 bad_schema=dict(valid);bad_schema['schema']='unknown-v9';malformed_cases.append(('wrong_schema',json.dumps(bad_schema).encode()))
 bad_type=dict(valid);bad_type['objects']=[dict(valid['objects'][0],absent=0)];malformed_cases.append(('wrong_field_type',json.dumps(bad_type).encode()))
 bad_key=dict(valid);bad_key['objects']=[dict(valid['objects'][0],key='../outside')];malformed_cases.append(('traversal_key',json.dumps(bad_key).encode()))
 duplicate=dict(valid);duplicate['objects']=valid['objects']*2;malformed_cases.append(('duplicate_key',json.dumps(duplicate).encode()))
 bad_digest=dict(valid);bad_digest['objects']=[dict(valid['objects'][0],sha256='xyz')];malformed_cases.append(('malformed_expected_digest',json.dumps(bad_digest).encode()))
 over_limit=dict(valid);over_limit['max_chunk_bytes']=32*1024*1024+1;malformed_cases.append(('configured_limit_over_hard_cap',json.dumps(over_limit).encode()))
 over_row=dict(valid);over_row['max_chunk_bytes']=32*1024*1024;over_row['objects']=[dict(valid['objects'][0],bytes=32*1024*1024+1)];malformed_cases.append(('row_over_hard_cap',json.dumps(over_row).encode()))
 for name,raw in malformed_cases:
  mpath=tests/(name+'.json');mpath.write_bytes(raw);case_env=dict(base_env,VC_VERIFIED_CT_MANIFEST=str(mpath),VC_VERIFIED_CT_MANIFEST_SHA256=hashlib.sha256(raw).hexdigest(),VC_VERIFIED_CT_LEDGER=str(tests/(name+'.jsonl')))
  r=subprocess.run([str(exe),str(group),'encoded'],env=case_env,text=True,capture_output=True,timeout=15)
  assert r.returncode==86 and 'mandatory evidence failure' in r.stderr,(name,r.returncode,r.stdout,r.stderr)
  rows.append({'case':'malformed_manifest_'+name,'returncode':r.returncode})
 for mode in ('store_stock','store_registry_stock'):
  r=subprocess.run([str(exe),str(root/'unrelated-store'),mode],env=plain,text=True,capture_output=True,timeout=15)
  assert r.returncode==0 and r.stdout.strip()=='abc',(mode,r.returncode,r.stdout,r.stderr)
  rows.append({'case':mode+'_works_without_opt_in','returncode':r.returncode})
 report={'status':'PASS_INTEGRATED_GUARDED_PUBLIC_ZARR_API','cases':rows,'scope':'Actual guarded libutils public API; no renderer or GPU; exact renderer I/O reconciliation remains required.'}
 (out/'integrated-fixture-report.json').write_text(json.dumps(report,indent=2));print(json.dumps({'status':report['status'],'cases':len(rows)}))
if __name__=='__main__':main()
