def main(num):
	if num > 100:
		print "bigger than 100"
	elif num > 0:
		print "between 0 to 100" 
	else:
		print "less than 100"

	a = range(100)
	print a

	for i in range(0, 100, 2):
		print i


	i = 100
	while i > 0:
		print i
		i -= 1

def foo(bar = []):
	if bar != []:
		bar = []

	bar.append("a")
	print bar



if __name__ == '__main__':
	integer = 10
	float_num = 102.1234
	strings = "hello World"
	arrays = [1, 2.3, "Think"]
	tuples = (1, 2, 3)
	dictionary = {"a": 2}
	sets = set([1, 2, 3, 3])

	# main(0)

	a = range(100)
	a = map(lambda x: x * 2, a)

	a = filter(lambda x: x % 8 == 0, a)
	a = reduce(lambda x, y: x + y, a)

	a = reduce(lambda x, y: x + y, [x * 2 for x in range(100) if (x * 2) % 8 == 0])
	print a

	foo()
	foo()