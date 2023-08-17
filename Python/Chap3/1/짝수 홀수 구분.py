# 입력
number = input("정수 입력")

# 마지막 자리 숫자 추출
last_char = number[-1]

# 숫자 변환
last_num = int(last_char)

# 짝수 확인
if last_num == 0 or  last_num == 2 or  last_num == 4 or  last_num == 6 or  last_num == 8:
    print("짝수")
# 홀수 확인
if last_num == 1 or  last_num == 3 or  last_num == 5 or  last_num == 7 or  last_num == 9:
    print("홀수")