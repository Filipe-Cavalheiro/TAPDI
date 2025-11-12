import pyopencl as cl
import numpy as np

def main():
    try:
        plaforms = cl.get_platforms()
        global plaform
        plaform = plaforms[0]
        devices = plaform.get_devices()
        global device
        device = devices[0]
        global ctx
        ctx = cl.Context(devices) # or dev_type=cl.device_type.ALL)
        global commQ
        commQ = cl.CommandQueue(ctx,device)
        file = open("C:\\Users\\caval\\Documents\\Universidade\\9_Semestre\\TAPDI\\TP5\\prog.cl","r")
        global prog
        prog = cl.Program(ctx,file.read())
        prog.build()
    except Exception as e:
        print(e)
        return False
    
    try:
        arrayIn = np.array(range(1,25), dtype=np.int32)
        kernelName = prog.power2
        memBuffer = cl.Buffer(ctx,
        flags= cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE, hostbuf=arrayIn)
        kernelName.set_arg( 0, memBuffer)
        globalWorkSize = (30,1)
        workGroupSize = (10,1)
        kernelEvent = cl.enqueue_nd_range_kernel( commQ, kernelName, 
        global_work_size= globalWorkSize, local_work_size= workGroupSize)
        kernelEvent.wait()
        cl.enqueue_copy(commQ, arrayIn, memBuffer)
        print(arrayIn)
        memBuffer.release()
    except Exception as e:
        print(e)
        return False
    
    try:
        result = np.zeros(1, dtype=np.int32)
        arrayIn = np.array(range(1,25), dtype=np.int32)
        kernelName = prog.power_const
        memBuffer = cl.Buffer(ctx, flags= cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE, hostbuf=arrayIn)
        return_memBuffer = cl.Buffer(ctx, flags= cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE, hostbuf=result)
        kernelName.set_arg( 0, memBuffer)
        kernelName.set_arg( 1, np.int32(5))
        kernelName.set_arg( 2, return_memBuffer)
        globalWorkSize = (30,1)
        workGroupSize = (10,1)
        kernelEvent = cl.enqueue_nd_range_kernel( commQ, kernelName, 
        global_work_size= globalWorkSize, local_work_size= workGroupSize)
        kernelEvent.wait()
        cl.enqueue_copy(commQ, result, return_memBuffer)
        print(result)
        memBuffer.release()
    except Exception as e:
        print(e)
        return False
    return True

main()