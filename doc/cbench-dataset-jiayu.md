位置： /home/jiayu/result_llvmtunerv3/
举例说明： /home/jiayu/result_llvmtunerv3/cBench/automotive_bitcount/random/result.json中保存着对automotive_bitcount的所有搜索结果，json文件的每行是一个list，第一个元素是pass序列，第二个元素是运行时间，每个样本对应的IR可以去samples文件夹下寻找，对应方式为：

假设 seq = "-ee-instrument -forceattrs -slsr"
那么 IR_id = hashlib.md5(seq.encode('utf-8')).hexdigest()
对应文件夹命名为： f"IR-{IR_id}"

最好结果及O3下运行时间：/home/jiayu/cBench_result/ （json文件中的最后一个数字代表O3下的运行时间）
