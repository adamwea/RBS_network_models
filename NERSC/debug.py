import pdb

def debug_function_wrapper():
    # Your code here
    from batchRun import main
    main()

if __name__ == "__main__":
    pdb.set_trace()
    try:
        debug_function_wrapper()
    except SyntaxError as e:
        if "Expecting value: line 1 column 1 (char 0)" in str(e):
            print("Encountered the specific error. Stopping program execution.")
            pdb.set_trace()
        else:
            raise