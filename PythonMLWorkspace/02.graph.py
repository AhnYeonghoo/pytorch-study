import random
import matplotlib
import matplotlib.pyplot as plt

# help(random)
# help(matplotlib)
# help(matplotlib.pyplot)

# plt.plot([1, 5, 7, 3, 7])
# plt.show()

month = ['mar', 'apr', 'may', 'jun', 'jul']
sales = [1, 5, 7, 3, 7]
plt.figure(dpi=150)
plt.rc('font', family='Malgun Gothic')
plt.title("월별 판매 실적")
plt.plot(month, sales, color='y', label='단위: 개수')
plt.legend(loc='lower right')
plt.xlabel('월')
plt.ylabel('매출')
plt.show()