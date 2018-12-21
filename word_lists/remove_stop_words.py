import string

f1 = open('word_lists/all_words_original.txt', 'r')
all_words = f1.read().split('\n')
f1.close()

f2 = open('word_lists/stop_words.txt', 'r')
stop_words = f2.read().split('\n')
f2.close()

for word in stop_words:
    if word in all_words:
        all_words.remove(word)

f3 = open("word_lists/all_words.txt", "w")
for word in all_words:
    f3.write(word)
    f3.write('\n')
# Remember to remove the final newline.