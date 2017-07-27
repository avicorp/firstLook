import cv2

from bankCheck import BankCheck


# check1 = BankCheck('../assets/Checks/1.png')
# check1.saveCheck('1.png')
# check1 = BankCheck('../assets/Checks/2.png')
# check1.saveCheck('2.png')
# check1 = BankCheck('../assets/Checks/3.png')
# check1.saveCheck('3.png')
# check1 = BankCheck('../assets/Checks/4.png')
# check1.saveCheck('4.png')
# check1 = BankCheck('../assets/Checks/5.png')
# check1.saveCheck('5.png')
# check1 = BankCheck('../assets/Checks/6.png')
# check1.saveCheck('6.png')
# check1 = BankCheck('../assets/Checks/7.png')
# check1.saveCheck('7.png')
# check1 = BankCheck('../assets/Checks/8.png')
# check1.saveCheck('8.png')

check1 = BankCheck('../assets/Checks/1.png')
check1.saveInputFields('checkInputFields1.png', True)
cv2.imwrite('amountFields1.png', check1.amountField())

check1 = BankCheck('../assets/Checks/2.png')
check1.saveInputFields('checkInputFields2.png', True)
cv2.imwrite('amountFields2.png', check1.amountField())
cv2.imwrite('dateFields2.png', check1.dateField())

check1 = BankCheck('../assets/Checks/3.png')
check1.saveInputFields('checkInputFields3.png', True)
cv2.imwrite('amountFields3.png', check1.amountField())
cv2.imwrite('dateFields3.png', check1.dateField())

check1 = BankCheck('../assets/Checks/4.png')
check1.saveInputFields('checkInputFields4.png', True)
cv2.imwrite('amountFields4.png', check1.amountField())
cv2.imwrite('dateFields4.png', check1.dateField())

check1 = BankCheck('../assets/Checks/5.png')
check1.saveInputFields('checkInputFields5.png', True)
cv2.imwrite('amountFields5.png', check1.amountField())
cv2.imwrite('dateFields5.png', check1.dateField())

check1 = BankCheck('../assets/Checks/6.png')
check1.saveInputFields('checkInputFields6.png', True)
cv2.imwrite('amountFields6.png', check1.amountField())
cv2.imwrite('dateFields6.png', check1.dateField())

check1 = BankCheck('../assets/Checks/7.png')
check1.saveInputFields('checkInputFields7.png', True)
cv2.imwrite('amountFields7.png', check1.amountField())
cv2.imwrite('dateFields7.png', check1.dateField())

check1 = BankCheck('../assets/Checks/8.png')
check1.saveInputFields('checkInputFields8.png', True)
cv2.imwrite('amountFields8.png', check1.amountField())
cv2.imwrite('dateFields8.png', check1.dateField())