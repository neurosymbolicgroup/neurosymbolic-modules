package org.genesys.interpreter;

import org.genesys.type.AbstractList;
import org.genesys.type.Cons;

/**
 * Created by yufeng on 6/4/17.
 */
public class ConsBinop implements Binop {
    public Object apply(Object first, Object second) {
        return new Cons(first, (AbstractList) second);
    }

    public String toString() {
        return "l(a,x).(cons a x)";
    }
}

