load "SYSTEM"
load "CONVERT"

REGISTER("slice", CREATE_BASIC_EXECUTOR("args[0][slice(*args[1:])]"))

func max(req a, req b){
    if (a > b){
        ret a
    }
    ret b
}

func min(req a, req b){
    if (a > b){
        ret b
    }
    ret a
}


func elementAt(req object, req index){
    if (index > length(object)){
        ret "NULL"
    }
    ret CALL_OBJECT(__getitem__ of object, index)
}

func toLower(req string){
    ret CALL_OBJECT(lower of SYSEXEC("str"), toString(string))
}