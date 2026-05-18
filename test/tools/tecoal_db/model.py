import datetime

class Case():
    def __init__(self):
        self.op_name = ""
        self.dtype = ""
        self.shape = ""
        self.layout = ""
        self.prototxt = ""
        self.case_path = ""
        self.hash = ""
        self.version = ""
        self.level = -1
        self.bug_id = -1
        self.baseline_data = ""
        self.baseline_type = ""
        self.baseline_time = ""
        self.status = 1
        self.count = 1

        # now = datetime.datetime.now()
        # self.create_at = now
        # self.updated_at = now
        # self.delete_at = now
        # self.test_time = now 

    def to_dict(self):
        return vars(self)


class Model():
    def __init__(self):
        self.name = ""
        self.version = ""
        self.image_size = ""
        self.status = 1

        # now = datetime.datetime.now()
        # self.create_at = now
        # self.updated_at = now
        # self.delete_at = now

    def to_dict(self):
        return vars(self)

    
class ModelCase():
    def __init__(self):
        self.model_id = -1
        self.case_id = -1
        self.count = 0

    def to_dict(self):
        return vars(self)


class Task():
    def __init__(self):
        self.tecoal = ""
        self.tecoblas = ""
        self.tecocustom = ""

        self.os = ""
        self.spe_clock = ""
        self.sdaadriver = ""
        self.sdaart = ""
        self.tecotest = ""

        self.test_type = -1
        self.status = 1
        
        # now = datetime.datetime.now()
        # self.create_at = now
        # self.updated_at = now
        # self.delete_at = now

        # case_count
        # pass_count

    def to_dict(self):
        return vars(self)

class TaskCase():
    def __init__(self):
        self.task_id = -1
        # self.case_id = -1

        self.op_name = ""
        self.dtype = ""
        self.shape = ""
        self.layout = ""
        self.case_path = ""
        self.hash = ""
        # self.version = ""
        # self.level = ""

        self.interface_time = -1
        self.hardware_time = -1
        self.min_hardware_time = -1
        self.max_hardware_time = -1
        self.minmax_hardware_time_gap = -1
        self.relative_minmax_hardware_time_gap = -1
        self.launch_time = -1
        self.io_bandwidth = -1
        self.compute_force = -1
        self.theory_ios = -1
        self.theory_ops = -1
        self.kernel_details = ""
        self.cache_miss_details = ""
        self.precision = ""
        self.gpu_precision = ""
        self.result = ""
        self.host_result = ""
        self.status = 1

        now = datetime.datetime.now()
        self.test_time = now.strftime("%Y-%m-%d %H:%M:%S") # now
        # self.create_at = now
        # self.updated_at = now
        # self.delete_at = now

    def to_dict(self):
        return vars(self)
