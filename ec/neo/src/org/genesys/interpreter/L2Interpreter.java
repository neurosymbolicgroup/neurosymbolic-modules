package org.genesys.interpreter;

import org.genesys.models.Node;
import org.genesys.models.Pair;
import org.genesys.type.AbstractList;
import org.genesys.type.Cons;
import org.genesys.type.EmptyList;
import org.genesys.type.Maybe;

import java.util.*;

/**
 * interpreter for L2 tool. Can be used in Deepcoder
 * Created by yufeng on 5/31/17.
 */
public class L2Interpreter extends BaseInterpreter {

    public L2Interpreter() {
        executors.put("apply_to_input", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(((Unop) objects.get(0)).apply(input));
            }
        });
        executors.put("true", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(true);
            }
        });
        executors.put("false", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(false);
            }
        });
        executors.put("0", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(0);
            }
        });
        executors.put("1", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(1);
            }
        });
        executors.put("+1", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>((int) objects.get(0) + 1);
            }
        });
        executors.put("-", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(-(int) objects.get(0));
            }
        });
        executors.put("emp", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new EmptyList());
            }
        });
        executors.put("cons", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new Cons(objects.get(0), (AbstractList) objects.get(1)));
            }
        });
        executors.put("pair", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new Pair(objects.get(0), objects.get(1)));
            }
        });
        executors.put("map", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new MapList((Unop) objects.get(0)));
            }
        });
        executors.put("filter", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new Filter((Unop) objects.get(0)));
            }
        });
        executors.put("foldLeft", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new FoldLeft((Binop) objects.get(0), objects.get(1)));
            }
        });
        executors.put("foldRight", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new FoldRight((Binop) objects.get(0), objects.get(1)));
            }
        });
        executors.put("l(a,x).(cons a x)", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new ConsBinop());
            }
        });
        executors.put("l(x,a).(cons a x)", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new ConsRevBinop());
            }
        });
        executors.put("l(a).(cons a x)", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new ConsFirstUnop((AbstractList) objects.get(0)));
            }
        });
        executors.put("l(x).(cons a x)", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new ConsSecondUnop(objects.get(0)));
            }
        });
        executors.put("l(a,b).(+ a b)", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new PrimitiveBinop("+"));
            }
        });
        executors.put("l(a,b).(* a b)", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new PrimitiveBinop("*"));
            }
        });
        executors.put("l(a,b).(% a b)", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new PrimitiveBinop("%"));
            }
        });
        executors.put("l(a,b).(> a b)", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new PrimitiveBinop(">"));
            }
        });
        executors.put("l(a,b).(< a b)", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new PrimitiveBinop("<"));
            }
        });
        executors.put("l(a,b).(>= a b)", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new PrimitiveBinop(">="));
            }
        });
        executors.put("l(a,b).(<= a b)", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new PrimitiveBinop("<="));
            }
        });
        executors.put("l(a,b).(== a b)", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new PrimitiveBinop("=="));
            }
        });
        executors.put("l(a,b).(|| a b)", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new PrimitiveBinop("||"));
            }
        });
        executors.put("l(a,b).(&& a b)", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new PrimitiveBinop("&&"));
            }
        });
        executors.put("l(a).(+ a b)", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new PrimitiveUnop("+", objects.get(0)));
            }
        });
        executors.put("l(a).(* a b)", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new PrimitiveUnop("*", objects.get(0)));
            }
        });
        executors.put("l(a).(> a b)", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new PrimitiveUnop(">", objects.get(0)));
            }
        });
        executors.put("l(a).(< a b)", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new PrimitiveUnop("<", objects.get(0)));
            }
        });
        executors.put("l(a).(>= a b)", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new PrimitiveUnop(">=", objects.get(0)));
            }
        });
        executors.put("l(a).(<= a b)", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new PrimitiveUnop("<=", objects.get(0)));
            }
        });
        executors.put("l(a).(== a b)", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new PrimitiveUnop("==", objects.get(0)));
            }
        });
        executors.put("l(a).(|| a b)", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new PrimitiveUnop("||", objects.get(0)));
            }
        });
        executors.put("l(a).(&& a b)", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new PrimitiveUnop("&&", objects.get(0)));
            }
        });
        executors.put("l(a).(- a)", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new PrimitiveUnop("-", null));
            }
        });
        executors.put("l(a).(~ a)", new Executor() {
            public Maybe<Object> execute(List<Object> objects, Object input) {
                return new Maybe<Object>(new PrimitiveUnop("~", null));
            }
        });
    }

}
