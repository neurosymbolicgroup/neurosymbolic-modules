package org.genesys.interpreter;

import org.genesys.type.AbstractList;
import org.genesys.type.Cons;

/**
 * Created by yufeng on 6/4/17.
 */
public class ConsFirstUnop implements Unop {
    private final AbstractList list;

    public ConsFirstUnop(AbstractList list) {
        this.list = list;
    }

    public Object apply(Object obj) {
        return new Cons(obj, this.list);
    }

    public String toString() {
        return "l(a).(cons a " + this.list.toString() + ")";
    }
}
