package org.genesys.interpreter.deepcode;

import org.genesys.interpreter.Unop;
import org.genesys.interpreter.Binop;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by yufeng on 5/31/17.
 */
public class CountList implements Unop {
    private final Binop op;
    private int rhs;

    public CountList(Binop unop, int l) {
        this.op = unop;
        this.rhs = l;
    }

    public Object apply(Object obj) {
        if (obj instanceof  Integer){
            assert ((Integer)obj == 256);
            return new ArrayList<>();
        }
        List list = (List) obj;
        int cnt = 0;
        if (list.isEmpty()) {
            return cnt;
        } else {
            for (Object elem : list) {
                if (op.toString().equals("l(a,b).(< a b)") && (Integer)elem < rhs) {
                    cnt++;
                } else if (op.toString().equals("l(a,b).(> a b)") && (Integer)elem > rhs) {
                    cnt++;
                } else if (op.toString().equals("l(a,b).(== a b)") && (Integer)elem == rhs) {
                    cnt++;
                } else if (op.toString().equals("l(a,b).(!= a b)") && (Integer)elem != rhs) {
                    cnt++;
                } else if (op.toString().equals("l(a,b).(%= a b)") && (Integer)elem % rhs == 0) {
                    cnt++;
                } else if (op.toString().equals("l(a,b).(%!= a b)") && (Integer)elem % rhs != 0) {
                    cnt++;
                }
            }
            return cnt;
        }
    }

    public String toString() {
        return "COUNT";
    }
}
