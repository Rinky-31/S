load "SYSTEM"
load "CONVERT"
load "ALGORITHM"

func createFileStream(req name, req spec){
    ret CALL_OBJECT(SYSEXEC("open"), name, spec)
}

func getOStream(req file){
    func write(text){
        ret CALL_OBJECT(write of file, toString(text))
    }
    ret write
}

func getIStream(req file){
    func readln(){
        text = CALL_OBJECT(readline of file)
        if (text contains SYSEXEC("'\n'")){
            text = slice(text, 0, -1, 1)
        }
        ret text
    }
    ret readln
}