load "SYSTEM"
load "CONVERT"
load "ALGORITHM"

REGISTER("createFileStream", 2, CREATE_BASIC_EXECUTOR("open(*args)"))
REGISTER("close", 1, CREATE_BASIC_EXECUTOR("args[0].close()"))

func getOStream(file){
    func write(text){
        ret CALL_OBJECT(ATTRIBUTE(file, "write"), toString(text))
    }
    ret write
}

func getIStream(file){
    func readln(){
        text = CALL_OBJECT(ATTRIBUTE(file, "readline"))
        if (contains(text, SYSEXEC("'\n'"))){
            text = slice(text, 0, -1, 1)
        }
        ret text
    }
    ret readln
}