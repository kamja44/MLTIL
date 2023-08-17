# in 연산자는 어떤 문자열 내부에 찾고자 하는 문자열이 있는지 확인할 때 사용한다./

# 입력
number = input("정수 입력")

# 마지막 자리 숫자 추출
last_char = number[-1]

if last_char in "13579":
    print("홀수")
if last_char in "02468":
    print("짝수")