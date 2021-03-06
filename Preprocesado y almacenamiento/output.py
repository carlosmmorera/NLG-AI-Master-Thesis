def printProgressBar (iteration, total, deleted_it, prefix = '', suffix = '', decimals = 2, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    delet = ''
    for k in deleted_it:
        delet += f' {k}: {deleted_it[k]}'
    print(f'\r{prefix} |{bar}| {percent}% {suffix} ({iteration}/{total}){delet}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def myprint(text, file_name = 'output.log'):
    with open(file_name, 'a') as f:
        f.write(text + '\n')