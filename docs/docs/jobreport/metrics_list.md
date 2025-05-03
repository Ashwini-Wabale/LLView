---
hide:
  - toc
---
# List of metrics

Here we list the current metrics and detail their meaning.

=== "Cores"

    * **Usage**  
    Average core usage over the runtime of the job per node (y-axis) and per core (x-axis) of the node. 
    
    !!! warning

        The abscissa in this graph are the **core IDs** instead of the **timestamp**, and it includes both "Physical" cores (first half) as well as the "Logical" ones (second-half) for multithreaded processors.

=== "CPU"

    * **CPU Usage**  
    1-min average usage of the CPU across all cores in a node. For multithreaded processors, the value can go up to 200% using physical and logical cores.

    * **Physical Cores Used**  
    Numbers of "Physical cores" with usage above 25% in the last minute. The "Physical cores" in the graphs are represented the first half of the node.

    * **Logical Cores Used** (For multithreaded processors)  
    Numbers of "Logical cores" with usage above 25% in the last minute. The "Logical cores" in the graphs are represented by the second half of the node.

    * **Load**  
    Average number of runnable processes (including those waiting for disk I/O) over the past 1 minute, indicating short-term system load and responsiveness (e.g., `1` means a load of 1 core on average - _not a percentage_).

    !!! note

        The Load is provided by Linux in three numbers: 1-, 5- and 15-min average loads. In the job reports, the `Node: Load` is obtained from Slurm, which at JSC contains the 1-min Load average.

    * **Memory Usage**  
    Amount of allocated RAM memory (in GiB) in the node.

    !!! note
        In the job reports, the `Node: Memory Usage` graphs (both for CPU and GPU) is scaled by default
        from 0 up to the memory limit of the partition. A swich between Job and System limits can be found on the interactive reports.

    !!! danger
        Some system processes may use up to a few GiB of memory on the system, so it is better to plan for 10-15GiB less than the maximum amount.

=== "GPU"

    * **Active SM**  
    Average fraction of time at least one warp was active on a multiprocessor, averaged over all multiprocessors.

    * **Utilization**  
    Percent of time over the past sample period during which one or more kernels was executing on the GPU.
      
    !!! warning

        The `Utilization` graph reflect the usage of at least one kernel on the GPU - it does not contain information of how much occupied it is. For this reason, it is recommended to check the `Active SM` metric described below.
      
    * **Memory Usage**  
    Amount of memory (in GiB) used on the device by the context.

    * **Temperature**  
    Current Temperature (in Celsius) on a given GPU. 
      
    !!! warning

        Note that high temperatures may trigger slow down of the GPU frequency (see examples of [High Temperature / GPU Throttling](examples.md#high-temperature-gpu-throttling)).
      
    * **Clk Throttle Reason**  
    Information about factors that are reducing the frequency of clocks. These are:
      
          ```
          1. GpuIdle - Nothing is running on the GPU and the clocks are dropping to Idle state.
          2. AppClkSet - GPU clocks are limited by applications clocks setting.
          3. SwPwrCap - SW Power Scaling algorithm is reducing the clocks below requested clocks because the GPU is consuming too much power.
          4. HWSlowDown - HW Slowdown (reducing the core clocks by a factor of 2 or more) is engaged. This is an indicator of:
                          * Temperature being too high
                          * External Power Brake Assertion is triggered (e.g. by the system power supply)
                          * Power draw is too high and Fast Trigger protection is reducing the clocks
          5.  SyncBoost - This GPU has been added to a Sync boost group with nvidia-smi or DCGM in order to maximize performance per watt. All GPUs in the sync boost group will boost to the minimum possible clocks across the entire group. Look at the throttle reasons for other GPUs in the system to see why those GPUs are holding this one at lower clocks.
          6.  SwThermSlDwn - SW Thermal Slowdown. This is an indicator of one or more of the following:
                             * Current GPU temperature above the GPU Max Operating Temperature
                             * Current memory temperature above the Memory Max Operating Temperature
          7.  HwThermSlDwn - HW Thermal Slowdown (reducing the core clocks by a factor of 2 or more) is engaged. This is an indicator of:
                             * Temperature being too high
          8.   PwrBrakeSlDwn - Power brake throttle to avoid that given racks draw more power than the facility can safely provide.
          ```

    !!! note

        The `Clk Throttle Reason` graphs are not shown when no throttling was ever active for the job.
      
    * **StreamMP Clk**  
    Current frequency in MHz of SM (Streaming Multiprocessor) clock. The frequency may be slowed down for the reasons given above.

    * **Memory Usage Rate**  
    Percent of time over the past sample period during which global (device) memory was being read or written. 

    * **Memory Clk**  
    Current frequency of the memory clock, in MHz.

    * **Performance State**  
    The current performance state for the GPU. States range from P0 (maximum performance) to P12 (minimum performance).
      
    !!! note

        The `Performance State` graphs are only shown when it differs from the default value of `0`.
      
    * **PCIE TX**  
    The GPU-centric transmission throughput across the PCIe bus (in GiB/s) over the past 20ms.

    * **PCIE RX**  
    The GPU-centric receive throughput across the PCIe bus (in GiB/s) over the past 20ms.

    !!! warning

        The `PCIE TX` and `PCIE RX` graphs only include throughput via PCIe bus, i.e., between GPU and CPU.

    * **NVLink TX**  
    The rate of data transmitted over NVLink in in GiB/s.

    * **NVLink RX**  
    The rate of data received over NVLink in GiB/s.

=== "Power"

    LLview can report power metrics (in Watts) at several levels:

    * **Node Power**  
    The total power draw for the entire node at the moment of sampling.

    !!! note
        "Node Power" values come from Slurm's `CurrentWatts` field (`scontrol show nodes`) and are snapshots taken once per minute. LLview may integrate these samples over time to estimate total energy consumption.

    * **CPU Power**  
    The instantaneous power consumed by the CPU package, including its memory controllers and system I/O.

    * **CPU Power Cap**  
    The enforced power limit on the CPU package. Displaying this is useful when users can modify CPU power caps or when they deviate from the system default.

    * **GPU Power**  
    The current power draw of each GPU device, including its onboard memory.

    * **Superchip Power**  
    On Grace–Hopper systems, LLview also reports the power usage for each “superchip” (i.e. combined Grace and Hopper modules).



=== "I/O (per File System)"

    * **Read**  
    Average read data rate (in MiB/s) in the last minute.

    * **Write**  
    Average write data rate (in MiB/s) in the last minute.

    * **Open/Close Operations**  
    Average operation rate (in operations/s) in the last minute.

=== "Interconnect"

    * **Data Input**  
    Average data input throughput (in MiB/s) in the last minute.

    * **Data Output**  
    Average data output throughput (in MiB/s) in the last minute.

    * **Packet Input**  
    Average package input throughput (in pkt/s) in the last minute.

    * **Packet Output**  
    Average package output throughput (in pkt/s) in the last minute.

    !!! attention

        The Interconnect values refer to input and output transfers to/from a given node, so it does not include communications within the node itself. However, I/O data is also included in the transferred data in or out of a node.

