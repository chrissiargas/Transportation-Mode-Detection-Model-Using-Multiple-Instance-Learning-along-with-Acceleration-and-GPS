from tensorboard import program
user = 1
tracking_address = r"C:\Users\chris\PycharmProjects\shlProject\logs_user" + str(user) + '/fullModelTb'  # the path of your log file.

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    tb.main()



