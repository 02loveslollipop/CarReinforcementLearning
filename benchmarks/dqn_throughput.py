import os,time,random,json,resource,io
import numpy as np, torch
from car_rl.environment import CarTrackEnv

torch.set_num_threads(1)
random.seed(1); np.random.seed(1); torch.manual_seed(1)
env=CarTrackEnv(track='circuit'); obs,_=env.reset(seed=1)
net=torch.nn.Sequential(torch.nn.Linear(10,128),torch.nn.ReLU(),torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,15)); target=__import__('copy').deepcopy(net); opt=torch.optim.Adam(net.parameters(),lr=0.0003)
N=20000; states=np.zeros((N,10),np.float32); nxt=np.zeros_like(states); actions=np.zeros(N,np.int64); rewards=np.zeros(N,np.float32); dones=np.zeros(N,np.float32)
steps=int(os.environ.get('STEPS','2000')); timing={'env':0.,'inference':0.,'update':0.}; episodes=0
start=time.perf_counter()
for i in range(steps):
 t=time.perf_counter()
 with torch.no_grad(): a=int(net(torch.from_numpy(obs)).argmax()) if i%10==0 else random.randrange(15)
 timing['inference']+=time.perf_counter()-t
 action=[(-1.,0.,1.)[a//5],(-1.,-.5,0.,.5,1.)[a%5]]
 t=time.perf_counter(); o,r,term,trunc,info=env.step(action); timing['env']+=time.perf_counter()-t
 k=i%N; states[k]=obs; nxt[k]=o; actions[k]=a; rewards[k]=r; dones[k]=term or trunc; obs=o
 if term or trunc: obs,_=env.reset(); episodes+=1
 if i>=256 and i%4==0:
  t=time.perf_counter(); idx=np.random.randint(0,min(i+1,N),64)
  q=net(torch.from_numpy(states[idx])).gather(1,torch.from_numpy(actions[idx,None])).squeeze(1)
  with torch.no_grad(): y=torch.from_numpy(rewards[idx])+0.99*(1-torch.from_numpy(dones[idx]))*target(torch.from_numpy(nxt[idx])).max(1).values
  loss=torch.nn.functional.smooth_l1_loss(q,y); opt.zero_grad(); loss.backward(); opt.step()
  if i%1000==0: target.load_state_dict(net.state_dict())
  timing['update']+=time.perf_counter()-t
elapsed=time.perf_counter()-start
start=time.perf_counter(); buf=io.BytesIO(); torch.save({'net':net.state_dict(),'target':target.state_dict(),'optimizer':opt.state_dict(),'states':states,'next':nxt,'actions':actions,'rewards':rewards,'dones':dones},buf); checkpoint=time.perf_counter()-start
print(json.dumps({'steps':steps,'seconds':elapsed,'steps_per_second':steps/elapsed,'timing':timing,'episodes':episodes,'checkpoint_serialize_seconds':checkpoint,'checkpoint_bytes':buf.tell(),'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}),flush=True)
