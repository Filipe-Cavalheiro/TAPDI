import pyopencl as cl
import python_files.imageForms as IF

platforms = cl.get_platforms()
for platform in platforms:
 name = platform.get_info(cl.platform_info.NAME)
 vendor = platform.get_info(cl.platform_info.VENDOR)
 version = platform.get_info(cl.platform_info.VERSION)
 displayStr = "Name: " + name + "\nVendor: " + vendor + "\nVersion: " + version + "\n"
 IF.showMessageBox(title="Platform Info", message=displayStr)
 devices = platform.get_devices()
 for device in devices:
  displayStr = "VENDOR: " + device.get_info(cl.device_info.VENDOR)
  displayStr = displayStr + "\nNAME: " + device.get_info(cl.device_info.NAME)
  displayStr = displayStr + "\nMAX_COMPUTE_UNITS: " + str(device.get_info(cl.device_info.MAX_COMPUTE_UNITS))
  displayStr = displayStr + "\nMAX_WORK_ITEM_DIMENSIONS: " + str(device.get_info(cl.device_info.MAX_WORK_ITEM_DIMENSIONS))
  displayStr = displayStr + "\nMAX_WORK_ITEM_SIZES: " + str(device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES))
  displayStr = displayStr + "\nMAX_WORK_GROUP_SIZE: " + str(device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE))
  displayStr = displayStr + "\nMAX_CONSTANT_ARGS: " + str(device.get_info(cl.device_info.MAX_CONSTANT_ARGS))
  displayStr = displayStr + "\nIMAGE_SUPPORT: " + str(device.get_info(cl.device_info.IMAGE_SUPPORT))
  displayStr = displayStr + "\nIMAGE2D_MAX_WIDTH: " + str(device.get_info(cl.device_info.IMAGE2D_MAX_WIDTH))
  displayStr = displayStr + "\nIMAGE2D_MAX_HEIGHT: " + str(device.get_info(cl.device_info.IMAGE2D_MAX_HEIGHT))
  displayStr = displayStr + "\nLOCAL_MEM_SIZE: " + str(device.get_info(cl.device_info.LOCAL_MEM_SIZE))
  displayStr = displayStr + "\nPREFERRED_WORK_GROUP_SIZE_MULTIPLE: " + str(device.get_info(cl.device_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE)) 
  IF.showMessageBox(title="Device Info", message=displayStr)