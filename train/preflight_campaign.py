"""CPU-only checks of the exact campaign images, before allocating training GPUs.

Run with the Modal SDK: python train/preflight_campaign.py
"""
from pathlib import Path
import modal

if modal.is_local():
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from modal_campaign import env_image,train_image,dashboard_image,ROOT
    images=[image.add_local_file(ROOT/'modal_campaign.py','/root/modal_campaign.py')
            for image in (env_image,train_image,dashboard_image)]
else:
    images=[None,None,None]
app=modal.App('kernel-env-cpu-startup-check')


@app.function(image=images[0],timeout=120,retries=0,max_containers=1,scaledown_window=2)
def evaluator_import():
    import modal_campaign
    from modal_app import _start_env_server
    from kernel_env.server.app import app
    assert callable(_start_env_server) and app
    print('evaluator runtime imports ok')


@app.function(image=images[1],timeout=180,retries=0,max_containers=1,scaledown_window=2)
def trainer_import():
    import sys
    sys.path.insert(0,'/root/train')
    import modal_campaign
    import campaign
    import grpo_kernel
    assert callable(campaign.main) and callable(grpo_kernel.main)
    print('trainer runtime imports ok')


@app.function(image=images[2],timeout=120,retries=0,max_containers=1,scaledown_window=2)
def dashboard_import():
    import modal_campaign
    from campaign_dashboard import make_app
    assert make_app(Path('/tmp/missing'),'test')
    print('dashboard runtime imports ok')


if __name__=='__main__':
    with modal.enable_output(),app.run():
        evaluator_import.remote()
        trainer_import.remote()
        dashboard_import.remote()
