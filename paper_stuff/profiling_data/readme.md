# Profiling Data

Based on the results, the barebones mujoco profiler which comes shipped with mujoco reports a simulation time per step of 117.9 us (see [mujoco_profiling.txt](./mujoco_profiling.txt)). When running cProfile on the env.step() call, (see [env_step.prof](env_step.prof)), we get that the env.step() takes 1749 us per call.

Remember that the env.step() runs are .05 seconds, and the mujoco step runs at .005 seconds, so each call to mj_step is actually called for 10 steps of simulation. From the mujoco profiler results, we'd expect 10 calls to take roughly 1180 us, which is very close to what we see in the cProfile results (1150 us). This means that the additional things like plugins and such are not causing much of slowdown. 

The extra stuff that slows down env.step() by 600us is the extra computation in reward computation all in python. Still though, the env.step() function is very fast. It has a realtime factor of .05 s/1749 us = 28.6, which is very good.