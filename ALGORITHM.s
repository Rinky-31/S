REGISTER("slice", 4, CREATE_BASIC_EXECUTOR("args[0][slice(*args[1:])]"))

func max(a, b){
    if (a > b){
        ret a
    }
    ret b
}

func min(a, b){
    if (a > b){
        ret b
    }
    ret a
}