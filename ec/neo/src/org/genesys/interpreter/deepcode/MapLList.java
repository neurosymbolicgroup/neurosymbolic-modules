package org.genesys.interpreter.deepcode;

import org.genesys.interpreter.Unop;
import org.genesys.type.AbstractList;
import org.genesys.type.Cons;
import org.genesys.type.EmptyList;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by yufeng on 5/31/17.
 */
public class MapLList implements Unop {
    private final Unop unop;

    public MapLList(Unop unop) {
        this.unop = unop;
    }

    public Object apply(Object obj) {
        if (obj instanceof  Integer){
            assert ((Integer)obj == 256);
            return new ArrayList<>();
        }
        List list = (List) obj;
        if (list.isEmpty()) {
            return list;
        } else {
            List targetList = new ArrayList();
            for(Object elem : list) {
                targetList.add(this.unop.apply(elem));
            }
            return targetList;
        }
    }

    public String toString() {
        return "MAP";
    }
}
