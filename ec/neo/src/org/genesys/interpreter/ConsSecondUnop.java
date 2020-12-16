package org.genesys.interpreter;

import org.genesys.type.AbstractList;
import org.genesys.type.Cons;

/**
 * Created by yufeng on 6/4/17.
 */
public class ConsSecondUnop implements Unop {
    private final Object obj;

    public ConsSecondUnop(Object obj) {
        this.obj = obj;
    }

    public Object apply(Object obj) {
        return new Cons(this.obj, (AbstractList) obj);
    }

    public String toString() {
        return "l(x).(cons " + this.obj.toString() + " x)";
    }
}
