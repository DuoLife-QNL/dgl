from prettytable import PrettyTable

# 创建一个 3x2 的表格
table = PrettyTable()
table.field_names = ["Name", "Age"]
table.add_row(["Alice", 25])
table.add_row(["Bob", 30])
table.add_row(["Charlie", 35])

# 将行和列进行转置
transposed_table = table.transpose()

# 打印原始表格和转置后的表格
print("Original table:")
print(table)
print("\nTransposed table:")
print(transposed_table)