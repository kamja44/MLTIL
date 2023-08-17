import datetime

now = datetime.datetime.now()

# 오전 구분
if now.hour < 12:
    print("현재 시간은 {}시 입니다. 오전입니다.".format(now.hour))

# 오후 구분
if now.hour >= 12:
    print("현재 시간은 {}시 입니다. 오후입니다.".format(now.hour))