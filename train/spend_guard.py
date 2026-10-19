"""One locked, shared ledger for every newly authorized job and retry."""
import fcntl
import json
import time
import uuid
from pathlib import Path


class SpendGuard:
    def __init__(self,path):self.path=Path(path)

    def update(self,operation):
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            ledger=json.loads(self.path.read_text())
            result=operation(ledger)
            temp=self.path.with_suffix('.tmp')
            temp.write_text(json.dumps(ledger,indent=2))
            temp.replace(self.path)
            return result

    def reserve(self,stage,envelope,seconds,gpus=2):
        if not 0<seconds<=6*3600:raise ValueError('invalid bounded job duration')
        if gpus not in (1,2):raise ValueError('invalid GPU count')
        def reserve(ledger):
            # $5/h per unpinned H100 bounds GPU + max8CPU/64GiB (~$4.84/h for one); retain startup/teardown time.
            amount=seconds*5*gpus/3600
            rows=ledger['new_jobs']
            if any(row['state'] in ('reserved','launched','uncertain','cancel_requested') for row in rows):
                raise RuntimeError('another authorized stage is still active or uncertain')
            used=sum(row['reserved_upper_usd'] for row in rows)
            group=sum(row['reserved_upper_usd'] for row in rows if row['envelope']==envelope)
            limit=ledger['envelopes_usd'][envelope]
            auxiliary=ledger.get('auxiliary_and_contingency_reserve_usd',10)
            evaluation_spent=sum(r['reserved_upper_usd'] for r in rows if r['envelope']=='evaluation')
            protected=max(0,ledger.get('protected_evaluation_usd',0)-evaluation_spent) if envelope!='evaluation' else 0
            if group+amount>limit+1e-9 or used+amount+auxiliary+protected>ledger['additional_cap_usd']+1e-9:
                raise RuntimeError('shared additional USD100 cap or stage envelope would be exceeded')
            item={'reservation_id':uuid.uuid4().hex,'stage':stage,'envelope':envelope,
                  'submitted_unix':time.time(),'deadline_unix':time.time()+seconds-30,
                  'reserved_upper_usd':amount,'max_resource_seconds':seconds,'gpus':gpus,'state':'reserved'}
            rows.append(item);ledger['new_reserved_upper_usd']=used+amount
            ledger['auxiliary_and_contingency_reserve_usd']=auxiliary
            return item
        return self.update(reserve)

    def record(self,reservation_id,**fields):
        def record(ledger):
            row=next(row for row in ledger['new_jobs'] if row['reservation_id']==reservation_id)
            row.update(fields)
            return row
        return self.update(record)
