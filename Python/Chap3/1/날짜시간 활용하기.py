# 날짜/시간과 관련된 기능을 가져온다
import datetime

# 현재 날짜/시간을 구한다.
now = datetime.datetime.now() # 현재 시간을 구해, now 변수에 대입한다.

# 출력한다.
print(now.year, "년")
print(now.month, "월")
print(now.day, "일")
print(now.hour, "시")
print(now.minute, "분")
print(now.second, "초")

# format 함수를 이용하여한 줄로 출력
print("{}년 {}월 {}일 {}시 {}분 {}초".format(now.year, now.month, now.day, now.hour, now.minute, now.second))