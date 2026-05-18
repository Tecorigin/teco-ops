# daily-test测试报告（{date}）

## 一、测试基本信息

* 测试环境

服务器{1}：

| module            | info                 |
| ----------------- | -------------------- |
| ip                | {ip}                 |
| sdaadriver        | {sdaadriver} |
| sdaart            | {sdaart}     |
| tecocc            | {tecocc}     |
| device(num)       | {device}         |
| device(spe clock) | {spe clock (MHz)}       |



* 测试对象

| module | version | change(diff commit num) |
|-----|-----|-----|
| tecoal |        {tecoal}           |                   |
| tecoblas |        {tecoblas}       |               |
| tecocustom |        {tecocustom}        |                |
| tecotest |        {tecotest}        |                |



## 二、测例统计

2.1 测例统计

| 测例数量   | 涵盖算子/算子总数            | 新增测例 | 删除测例 |
| ---------- | ---------------------------- | -------- | -------- |
| {tot_cases} | {tot_ops}/{total_op_num} |     {new_cases_num}     |   {delete_cases_num}       |




### 2.1 测例变化统计表
|op_name|新增|删除|
|-----|-----|-----|
{op_cases_change}




## 三、测试结果

详细测试结果可以见表：{xlsx_path}

### 3.1 监管测例

#### 1) 精度测试结果
| 测试通过百分比 | 测试通过数量 | bug修复   | <span style='color: black; font-weight: bold;'>**新增bug**</span> | <span style='color: black; font-weight: bold;'>**未修复bug**</span> |
| -------------- | ------------ | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| {pass_per}     | {tot_passed}   | {fix_num} | {new_bug_num}                                                | {unfix_num}                                                  |

bug修复涉及算子：{fix_op}

新增bug涉及算子：{new_bug_op}

未修复bug涉及算子：{unfix_op}

精度提升算子(对比前一天): 
{accu_rise_day_op}

精度提升算子(对比上个版本): 
{accu_rise_version_op}

精度下降算子(对比前一天): 
{accu_fallback_day_op}

精度下降算子(对比上个版本): 
{accu_fallback_version_op}

#### 2) 性能测试结果（对比上一个版本）

（精度测试通过的测例）

| 测例数量 | 性能提升 | 性能下降 | 性能持平 |
| -------- | -------- | -------- | -------- |
{cmp_with_version_summary}



性能提升



<div style="width:6%">  </div>|<div style="width:12%">op</div>|<div style="width:12%">性能提升最大百分比</div> |<div style="width:12%"> pre_hardware_time(ms)</div> | <div style="width:12%">cur_hardware_time(ms)</div> |<div style="width:23%"> pre_kernel</div> | <div style="width:23%">cur_kernel</div> |
| ---- | ---- | ------------------ | -------------------------- | ------------------------- | --------------- | -------------- |
{cmp_with_version_rise}



性能下降


<div style="width:6%">  </div>|<div style="width:12%">op</div>|<div style="width:12%">性能回退最大百分比</div> |<div style="width:12%"> pre_hardware_time(ms)</div> | <div style="width:12%">cur_hardware_time(ms)</div> |<div style="width:23%"> pre_kernel</div> | <div style="width:23%">cur_kernel</div> |
| ---- | ---- | ------------------ | -------------------------- | ------------------------- | --------------- | -------------- |
{cmp_with_version_fallback}



#### 3) 性能测试结果（对比前一日版本）

（精度测试通过的测例）

| 测例数量 | 性能提升 | 性能下降 | 性能持平 |
| -------- | -------- | -------- | -------- |
{cmp_with_day_summary}



性能提升


<div style="width:6%">  </div>|<div style="width:12%">op</div>|<div style="width:12%">性能提升最大百分比</div> |<div style="width:12%"> pre_hardware_time(ms)</div> | <div style="width:12%">cur_hardware_time(ms)</div> |<div style="width:23%"> previous_kernel</div> | <div style="width:23%">current_kernel</div> |
| ---- | ---- | ------------------ | -------------------------- | ------------------------- | --------------- | -------------- |
{cmp_with_day_rise}



性能下降


<div style="width:6%">  </div>|<div style="width:12%">op</div>|<div style="width:12%">性能回退最大百分比</div> |<div style="width:12%"> pre_hardware_time(ms)</div> | <div style="width:12%">cur_hardware_time(ms)</div> |<div style="width:23%"> pre_kernel</div> | <div style="width:23%">cur_kernel</div> |
| ---- | ---- | ------------------ | -------------------------- | ------------------------- | --------------- | -------------- |
{cmp_with_day_fallback}




### 3.2 新增测例

#### 1)精度结果

|       | 算子     | 新增case数量     | 测试通过数量  | 测试失败数量（包含精度不达标） | <span style='color: black; font-weight: bold;'>**精度不达标数量**</span> | 最大带宽/最小带宽     | 最大计算效率/最小计算效率   |
| ----- | -------- | ---------------- | ------------- | ------------------------------ | ------------------------------------------------------------ | --------------------- | --------------------------- |
{add_op_details}
|total|{tot_add_details}|||