import time,json,resource,numpy as np,torch
from car_rl.environment import CarTrackEnv
torch.set_num_threads(1); torch.manual_seed(1)
net=torch.nn.Sequential(torch.nn.Linear(10,128),torch.nn.ReLU(),torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,15))
t=time.perf_counter(); steps=0; lengths=[]
for track in ['circuit','oval','clover']:
 env=CarTrackEnv(track=track)
 for frac in [0,.25,.5,.75]:
  obs,_=env.reset(seed=0,options={'s0':env.track.length*frac}); done=False; count=0
  while not done:
   with torch.no_grad(): a=int(net(torch.from_numpy(obs)).argmax())
   obs,_,term,trunc,_=env.step([(-1.,0.,1.)[a//5],(-1.,-.5,0.,.5,1.)[a%5]])
   count+=1; done=term or trunc
  steps+=count; lengths.append(count)
print(json.dumps({'evaluation_seconds':time.perf_counter()-t,'steps':steps,'episode_lengths':lengths,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))
