import datetime
from db import db, app, session
from sqlalchemy import UniqueConstraint
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import Insert


@compiles(Insert)
def prefix_inserts(insert, compiler, **kw):
    s = compiler.visit_insert(insert.prefix_with("IGNORE"), **kw)
    return s


# @compiles(Insert)
# def append_string(insert, compiler, **kw):
#     s = compiler.visit_insert(insert, **kw)
#     print(s)
#     if 'append_string' in insert.kwargs:
#         return s + " " + insert.kwargs['append_string']
#     return s

# my_connection.execute(my_table.insert(append_string='ON DUPLICATE KEY'),
#                       my_values)


class Case(db.Model):
    __tablename__ = 'tecoal_test_case'
    id = db.Column(db.Integer, primary_key=True)
    op_name = db.Column(db.String(255), default="")
    dtype = db.Column(db.String(255), default="")
    shape = db.Column(db.String(255), default="")
    layout = db.Column(db.String(255), default="")
    prototxt = db.Column(db.Text, default="")
    case_path = db.Column(db.String(500), default="", unique=True)
    hash = db.Column(db.String(255), default="")

    version = db.Column(db.String(255), default="")
    level = db.Column(db.Integer, default=-1)

    bug_id = db.Column(db.Integer, default=-1)

    create_at = db.Column(db.Time)
    updated_at = db.Column(db.Time)
    deleted_at = db.Column(db.Time)

    baseline_data = db.Column(db.Text(), default="")
    baseline_type = db.Column(db.String(255), default="")
    baseline_time = db.Column(db.String(255), default="")
    test_time = db.Column(db.Time)

    status = db.Column(db.Integer, default=-1)  # 1: 使用 2：不使用
    count = db.Column(db.Integer, default=-1)

    def __init__(self):
        now = datetime.datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
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

        self.create_at = now
        self.updated_at = now
        self.deleted_at = now

        self.baseline_data = ""
        self.baseline_type = ""
        self.baseline_time = ""
        self.test_time = now

        self.status = 1

        self.count = 1  # 不加入数据库

    def to_dict(self):
        res = {
            c.name: getattr(self, c.name, None)
            for c in self.__table__.columns
        }
        for key in ["id", "create_at", "updated_at", "deleted_at", "test_time"]:
            if key in res:
                del res[key]
        return res

    def add(self, cases):
        if type(cases) == list:
            session.add_all(cases)
        else:
            session.add(cases)
        session.commit()

    # def delete(self, id):
    #     case = Case.query.get(id)
    #     case.enable = 2
    #     session.commit()

    # def update(self, id, **kwargs):
    #     case = Case.query.get(id)
    #     for key, value in kwargs.items():
    #         if hasattr(case, key):
    #             setattr(case, key, value)
    #     session.commit()

    def query(self, **kwargs):
        res = session.query(Case).filter_by(**kwargs).all()
        return res

    # # todo 自定义查询
    # def query(self, data=None):
    #     if data == None:
    #         cases = Case.query.all()
    #     else:
    #         cases = Case.query.filter_by(data)
    #     return cases


class Model(db.Model):
    __tablename__ = 'tecoal_test_model'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), default="", unique=True)
    image_size = db.Column(db.String(255), default="")

    version = db.Column(db.String(255), default="")
    create_at = db.Column(db.Time)
    updated_at = db.Column(db.Time)
    deleted_at = db.Column(db.Time)
    status = db.Column(db.Integer, default=-1)

    def __init__(self):
        now = datetime.datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
        self.name = ""
        self.image_size = ""

        self.version = ""
        self.create_at = now
        self.updated_at = now
        self.deleted_at = now

        self.status = 1

    def to_dict(self):
        res = {
            c.name: getattr(self, c.name, None)
            for c in self.__table__.columns
        }
        for key in ["id", "create_at", "updated_at", "deleted_at", "test_time"]:
            if key in res:
                del res[key]
        return res

    def add(self, models):
        if type(models) == list:
            session.add_all(models)
        else:
            session.add(models)
        session.commit()

    def query(self, **kwargs):
        res = session.query(Model).filter_by(**kwargs).all()
        return res


class ModelCase(db.Model):
    __tablename__ = 'tecoal_test_model_case'
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, default=-1)
    case_id = db.Column(db.Integer, default=-1)
    count = db.Column(db.Integer, default=-1)
    __table_args__ = (UniqueConstraint('model_id', 'case_id'), )

    def __init__(self):
        self.model_id = -1
        self.case_id = -1
        self.count = 0

    def to_dict(self):
        res = {
            c.name: getattr(self, c.name, None)
            for c in self.__table__.columns
        }
        for key in ["id", "create_at", "updated_at", "deleted_at"]:
            if key in res:
                del res[key]
        return res

    def add(self, model_cases):
        if type(model_cases) == list:
            session.add_all(model_cases)
        else:
            session.add(model_cases)
        session.commit()


class Task(db.Model):
    __tablename__ = 'tecoal_test_task'
    id = db.Column(db.Integer, primary_key=True)
    tecoal = db.Column(db.String(255), default="")
    tecoblas = db.Column(db.String(255), default="")
    tecocustom = db.Column(db.String(255), default="")

    os = db.Column(db.String(255), default="")
    spe_clock = db.Column(db.String(255), default="")
    sdaadriver = db.Column(db.String(255), default="")
    sdaart = db.Column(db.String(255), default="")
    tecotest = db.Column(db.String(255), default="")

    test_type = db.Column(db.Integer, default=-1)
    status = db.Column(db.Integer, default=1)

    create_at = db.Column(db.Time)
    updated_at = db.Column(db.Time)
    deleted_at = db.Column(db.Time)
    # case_count
    # pass_count

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
        
        now = datetime.datetime.now()
        self.create_at = now
        self.updated_at = now
        self.deleted_at = now

    def to_dict(self):
        res = {
            c.name: getattr(self, c.name, None)
            for c in self.__table__.columns
        }
        for key in ["id", "create_at", "updated_at", "deleted_at"]:
            if key in res:
                del res[key]
        return res

    def query(self, **kwargs):
        res = session.query(Task).filter_by(**kwargs).all()
        return res

    def __str__(self):
        return f"{self.os=}, {self.spe_clock=}, {self.tecoal=}, {self.tecoblas=}, {self.tecocustom=}, {self.sdaadriver=}, {self.sdaart=}"

    def add(self, tasks):
        if type(tasks) == list:
            session.add_all(tasks)
        else:
            session.add(tasks)
        session.commit()


class TaskCase(db.Model):
    __tablename__ = 'tecoal_test_task_case'
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.Integer, default=-1)
    # case_id = db.Column(db.Integer, default=-1)

    op_name = db.Column(db.String(255), default="")
    dtype = db.Column(db.String(255), default="")
    shape = db.Column(db.String(255), default="")
    layout = db.Column(db.String(255), default="")
    case_path = db.Column(db.String(500), default="", unique=True)
    # hash version level

    interface_time = db.Column(db.Float, default=-1)
    hardware_time = db.Column(db.Float, default=-1)
    min_hardware_time = db.Column(db.Float, default=-1)
    max_hardware_time = db.Column(db.Float, default=-1)
    minmax_hardware_time_gap = db.Column(db.Float, default=-1)
    relative_minmax_hardware_time_gap = db.Column(db.Float, default=-1)
    launch_time = db.Column(db.Float, default=-1)
    io_bandwidth = db.Column(db.Float, default=-1)
    compute_force = db.Column(db.Float, default=-1)
    theory_ios = db.Column(db.Float, default=-1)
    theory_ops = db.Column(db.Float, default=-1)
    kernel_details = db.Column(db.String(1000), default="")
    cache_miss_details = db.Column(db.String(1000), default="")
    precision = db.Column(db.Text, default="")
    gpu_precision = db.Column(db.Text, default="")

    result = db.Column(db.String(255), default="")
    host_result = db.Column(db.String(255), default="")
    test_time = db.Column(db.String(255), default="") # db.Column(db.Time)

    status = db.Column(db.Integer, default=1)

    create_at = db.Column(db.Time)
    updated_at = db.Column(db.Time)
    deleted_at = db.Column(db.Time)

    def __init__(self):
        now = datetime.datetime.now()

        self.task_id = -1
        # self.case_id = -1

        self.op_name = ""
        self.dtype = ""
        self.shape = ""
        self.layout = ""
        self.case_path = ""

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
        self.test_time = now.strftime("%Y-%m-%d %H:%M:%S") # now

        self.status = 1

        self.create_at = now
        self.updated_at = now
        self.deleted_at = now

    def to_dict(self):
        res = {
            c.name: getattr(self, c.name, None)
            for c in self.__table__.columns
        }
        for key in ["id", "create_at", "updated_at", "deleted_at"]:
            if key in res:
                del res[key]
        return res

    def add(self, task_cases):
        if type(task_cases) == list:
            session.add_all(task_cases)
        else:
            session.add(task_cases)
        session.commit()

class ModelOpPerformance(db.Model):
    __tablename__ = 'tecoal_test_model_op_performance'
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, default=-1)
    opname = db.Column(db.String(255), default=-1)
    performance = db.Column(db.Float, default=-1)

    def __init__(self):
        self.model_id = -1
        self.opname = ""
        self.performance = -1

    def to_dict(self):
        res = {
            c.name: getattr(self, c.name, None)
            for c in self.__table__.columns
        }
        for key in ["id", "create_at", "updated_at", "deleted_at"]:
            if key in res:
                del res[key]
        return res

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
