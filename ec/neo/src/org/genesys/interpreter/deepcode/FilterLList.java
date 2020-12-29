package org.genesys.interpreter.deepcode;

import org.genesys.interpreter.Unop;
import org.genesys.interpreter.Binop;
import org.genesys.type.AbstractList;
import org.genesys.type.Cons;
import org.genesys.type.EmptyList;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by yufeng on 5/31/17.
 */
public class FilterLList implements Unop {
    private final Binop op;

    private Integer rhs;

    public FilterLList(Binop op, int l) {
        this.op = op;
        this.rhs = l;
    }

    public Object apply(Object obj) {
        if (obj instanceof  Integer){
            assert ((Integer)obj == 256);
            return new ArrayList<>();
        }
        List list = (List)  obj;
        if (list.isEmpty()) {
            return list;
        } else {
            List targetList = new ArrayList<>();
            for (Object elem : list) {
                if (op.toString().equals("l(a,b).(< a b)") && (Integer)elem < rhs) {
                    targetList.add(elem);
                } else if (op.toString().equals("l(a,b).(> a b)") && (Integer)elem > rhs) {
                    targetList.add(elem);
                } else if (op.toString().equals("l(a,b).(== a b)") && (Integer)elem == rhs) {
                    targetList.add(elem);
                } else if (op.toString().equals("l(a,b).(!= a b)") && (Integer)elem != rhs) {
                    targetList.add(elem);
                } else if (op.toString().equals("l(a,b).(%= a b)")){
                    if (rhs == 0)
                        return new ArrayList<>();
                    else if((Integer)elem % rhs == 0){
                        targetList.add(elem);
                    }
                } else if (op.toString().equals("l(a,b).(%!= a b)")){
                    if (rhs == 0)
                        return new ArrayList<>();
                    else if((Integer)elem % rhs != 0){
                        targetList.add(elem);
                    }
                }
            }
            return targetList;
        }
    }

    public String toString() {
        return "FILTER";
    }
}
