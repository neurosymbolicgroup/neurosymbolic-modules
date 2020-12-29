package org.genesys.language;

import org.genesys.type.*;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by yufeng on 5/31/17.
 * Directly use the grammar from Osbert.
 */
public class L2Grammar implements Grammar<AbstractType> {

    public AbstractType inputType;

    public AbstractType outputType;

    public L2Grammar(AbstractType inputType, AbstractType outputType) {
        this.inputType = inputType;
        this.outputType = outputType;
    }

    @Override
    public AbstractType start() {
        return new AppToInputType(new FunctionType(this.inputType, this.outputType));
    }

    @Override
    public AbstractType getOutputType(){ return outputType; }

    @Override
    public String getName() {
        return "L2Grammar";
    }

    @Override
    public List<Production<AbstractType>> getProductions() {
        return null;
    }

    @Override
    public List<Production<AbstractType>> getInputProductions() { return null; }


    @Override
    public List<Production<AbstractType>> getLineProductions(int size) { return null; }

    @Override
    public List<Production<AbstractType>> productionsFor(AbstractType symbol) {
        List<Production<AbstractType>> productions = new ArrayList<>();
        if (symbol instanceof AppToInputType) {
            AppToInputType type = (AppToInputType) symbol;
            productions.add(new Production<>(symbol, "apply_to_input", type.functionType));
        } else if (symbol instanceof BoolType) {
            productions.add(new Production<>(symbol, "true"));
            productions.add(new Production<>(symbol, "false"));
        } else if (symbol instanceof IntType) {
            productions.add(new Production<>(symbol, "0"));
            productions.add(new Production<>(symbol, "1"));
            productions.add(new Production<>(symbol, "+1", symbol));
            productions.add(new Production<>(symbol, "-", new IntType()));
        } else if (symbol instanceof ListType) {
            ListType type = (ListType) symbol;
            productions.add(new Production<>(symbol, "emp"));
            productions.add(new Production<>(symbol, "cons", type.type, symbol));
        } else if (symbol instanceof PairType) {
            PairType type = (PairType) symbol;
            productions.add(new Production<>(symbol, "pair", type.firstType, type.secondType));
        } else if (symbol instanceof FunctionType) {
            FunctionType type = (FunctionType) symbol;
            // map (I -> O) ::= (List<I> -> List<O>)
            if (type.inputType instanceof ListType && type.outputType instanceof ListType) {
                AbstractType I = ((ListType) type.inputType).type;
                AbstractType O = ((ListType) type.outputType).type;
                productions.add(new Production<>(symbol, "map", new FunctionType(I, O)));
            }
            // filter (T -> Boolean) ::= (List<T> -> List<T>)
            if (type.inputType instanceof ListType && type.outputType instanceof ListType
                    && type.inputType.equals(type.outputType)) {
                AbstractType T = ((ListType) type.inputType).type;
                productions.add(new Production<>(symbol, "filter", new FunctionType(T, new BoolType())));
            }
            // fold ((I, O) -> O, O) ::= (List<I> -> O)
            if (type.inputType instanceof ListType) {
                AbstractType I = ((ListType) type.inputType).type;
                AbstractType O = type.outputType;
                productions.add(new Production<>(symbol, "foldLeft", new FunctionType(new PairType(I, O), O), O));
                productions.add(new Production<>(symbol, "foldRight", new FunctionType(new PairType(I, O), O), O));
            }
            // l(a,x).(cons a x) ::= ((T, List<T>) -> List<T>)
            if (type.inputType instanceof PairType && ((PairType) type.inputType).secondType instanceof ListType
                    && type.outputType instanceof ListType) {
                AbstractType T = ((ListType) ((PairType) type.inputType).secondType).type;
                if (T.equals(((PairType) type.inputType).firstType) && T.equals(((ListType) type.outputType).type)) {
                    productions.add(new Production<>(symbol, "l(a,x).(cons a x)"));
                }
            }
            // l(x,a).(cons a x) ::= ((List<T>, T) -> List<T>)
            if (type.inputType instanceof PairType && ((PairType) type.inputType).firstType instanceof ListType
                    && type.outputType instanceof ListType) {
                AbstractType T = ((ListType) ((PairType) type.inputType).firstType).type;
                if (T.equals(((PairType) type.inputType).secondType) && T.equals(((ListType) type.outputType).type)) {
                    productions.add(new Production<>(symbol, "l(x,a).(cons a x)"));
                }
            }
            // l(a).(cons a x) (List<T>) ::= (T -> List<T>)
            if (type.outputType instanceof ListType && type.inputType.equals(((ListType) type.outputType).type)) {
                AbstractType T = type.outputType;
                productions.add(new Production<>(symbol, "l(a).(cons a x)", T));
            }
            // l(x).(cons a x) (T) ::= (List<T> -> List<T>)
            if (type.inputType instanceof ListType && type.outputType instanceof ListType
                    && type.inputType.equals(type.outputType)) {
                AbstractType T = ((ListType) type.inputType).type;
                productions.add(new Production<>(symbol, "l(x).(cons a x)", T));
            }
            // l(a,b).(+ a b) ::= ((Integer, Integer) -> Integer)
            // l(a,b).(* a b) ::= ((Integer, Integer) -> Integer)
            if (type.inputType instanceof PairType && ((PairType) type.inputType).firstType instanceof IntType
                    && ((PairType) type.inputType).secondType instanceof IntType && type.outputType instanceof IntType) {
                productions.add(new Production<>(symbol, "l(a,b).(+ a b)"));
                productions.add(new Production<>(symbol, "l(a,b).(* a b)"));
                productions.add(new Production<>(symbol, "l(a,b).(% a b)"));
            }
            // l(a,b).(> a b) ::= ((Integer, Integer) -> Boolean)
            // l(a,b).(< a b) ::= ((Integer, Integer) -> Boolean)
            // l(a,b).(>= a b) ::= ((Integer, Integer) -> Boolean)
            // l(a,b).(<= a b) ::= ((Integer, Integer) -> Boolean)
            if (type.inputType instanceof PairType && ((PairType) type.inputType).firstType instanceof IntType
                    && ((PairType) type.inputType).secondType instanceof IntType && type.outputType instanceof BoolType) {
                productions.add(new Production<>(symbol, "l(a,b).(> a b)"));
                productions.add(new Production<>(symbol, "l(a,b).(< a b)"));
                productions.add(new Production<>(symbol, "l(a,b).(>= a b)"));
                productions.add(new Production<>(symbol, "l(a,b).(<= a b)"));
                productions.add(new Production<>(symbol, "l(a,b).(== a b)"));
            }
            // l(a,b).(|| a b) ::= ((Boolean, Boolean) -> Boolean)
            // l(a,b).(&& a b) ::= ((Boolean, Boolean) -> Boolean)
            if (type.inputType instanceof PairType && ((PairType) type.inputType).firstType instanceof BoolType
                    && ((PairType) type.inputType).secondType instanceof BoolType && type.outputType instanceof BoolType) {
                productions.add(new Production<>(symbol, "l(a,b).(|| a b)"));
                productions.add(new Production<>(symbol, "l(a,b).(&& a b)"));
            }
            // l(a).(+ a b) (Integer) ::= (Integer -> Integer)
            // l(a).(* a b) (Integer) ::= (Integer -> Integer)
            if (type.inputType instanceof IntType && type.outputType instanceof IntType) {
                productions.add(new Production<>(symbol, "l(a).(+ a b)", new IntType()));
                productions.add(new Production<>(symbol, "l(a).(* a b)", new IntType()));
                productions.add(new Production<>(symbol, "l(a).(% a b)", new IntType()));
            }
            // l(a).(> a b) (Integer) ::= (Integer -> Boolean)
            // l(a).(< a b) (Integer) ::= (Integer -> Boolean)
            // l(a).(>= a b) (Integer) ::= (Integer -> Boolean)
            // l(a).(<= a b) (Integer) ::= (Integer -> Boolean)
            if (type.inputType instanceof IntType && type.outputType instanceof BoolType) {
                productions.add(new Production<>(symbol, "l(a).(> a b)", new IntType()));
                productions.add(new Production<>(symbol, "l(a).(< a b)", new IntType()));
                productions.add(new Production<>(symbol, "l(a).(>= a b)", new IntType()));
                productions.add(new Production<>(symbol, "l(a).(<= a b)", new IntType()));
                productions.add(new Production<>(symbol, "l(a).(== a b)", new IntType()));
            }
            // l(a).(|| a b) (Integer) ::= (Boolean -> Boolean)
            // l(a).(&& a b) (Integer) ::= (Boolean -> Boolean)
            if (type.inputType instanceof PairType && ((PairType) type.inputType).firstType instanceof BoolType
                    && ((PairType) type.inputType).secondType instanceof BoolType && type.outputType instanceof BoolType) {
                productions.add(new Production<>(symbol, "l(a).(|| a b)", new BoolType()));
                productions.add(new Production<>(symbol, "l(a).(&& a b)", new BoolType()));
            }
            // l(a).(- a) ::= (Integer -> Integer)
            if (type.inputType instanceof IntType && type.outputType instanceof IntType) {
                productions.add(new Production<>(symbol, "l(a).(- a)"));
            }
            // l(a).(~ a) ::= (Boolean -> Boolean)
            if (type.inputType instanceof BoolType && type.outputType instanceof BoolType) {
                productions.add(new Production<>(symbol, "l(a).(~ a)"));
            }
        }
        return productions;
    }
}
