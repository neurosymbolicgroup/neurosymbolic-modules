package org.genesys.interpreter;

import org.genesys.type.AbstractList;
import org.genesys.type.Cons;

/**
 * Created by yufeng on 6/4/17.
 */
public class ConsRevBinop implements Binop {

    public Object apply(Object first, Object second) {
        return new Cons(second, (AbstractList) first);
    }

    public String toString() {
        return "l(x,a).(cons a x)";
    }
}
