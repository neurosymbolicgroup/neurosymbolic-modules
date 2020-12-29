package org.genesys.type;

import java.util.LinkedList;
import java.util.List;

/**
 * FIXME: Should Cons be a component instead of a type?
 * Created by yufeng on 5/31/17.
 */
public class Cons implements AbstractList {
    public Object obj;
    public AbstractList list;

    public Cons(Object obj, AbstractList list) {
        this.obj = obj;
        this.list = list;
    }

    @Override
    public String toString() {
        return "(cons " + this.obj.toString() + " " + this.list.toString() + ")";
    }
}
