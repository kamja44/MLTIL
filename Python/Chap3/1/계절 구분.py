import datetime

now = datetime.datetime.now()

# 봄 구분
if 3 <= now.month <= 5:
    print("이번 달은 {}월입니다. 봄입니다.".format(now.month))

# 여름 구분
if 6 <= now.month <= 8:
    print("이번 달은 {}월입니다. 여름입니다.".format(now.month))

# 가을 구분
if 9 <= now.month <= 11:
    print("이번 달은 {}월입니다. 가을입니다.".format(now.month))

# 겨울 구분
if 12 <= now.month <= 2:
    print("이번 달은 {}월입니다. 겨울입니다.".format(now.month))
