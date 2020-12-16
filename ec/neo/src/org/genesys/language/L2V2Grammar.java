package org.genesys.language;

import org.genesys.models.Example;
import org.genesys.models.Problem;
import org.genesys.type.*;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by yufeng on 5/31/17.
 * Grammar for L2 in deepcode style.
 */
public class L2V2Grammar implements Grammar<AbstractType> {

    public AbstractType inputType;

    public AbstractType outputType;

    private List<InputType> inputTypes = new ArrayList<>();

    public L2V2Grammar(AbstractType inputType, AbstractType outputType) {
        this.inputType = inputType;
        inputTypes.add(new InputType(0, inputType));
        this.outputType = outputType;
    }

    public L2V2Grammar(Problem p) {
        assert !p.getExamples().isEmpty();
        Example example = p.getExamples().get(0);
        List input = example.getInput();
        for (int i = 0; i < input.size(); i++) {
            Object elem = input.get(i);
            InputType in;
            if (elem instanceof List)
                in = new InputType(i, new ListType(new IntType()));
            else
                in = new InputType(i, new IntType());

        /* dynamically add input to grammar. */
            addInput(in);
        }
        Object output = example.getOutput();
        //output is either an integer or list.
        if (output instanceof List) {
            this.outputType = new ListType(new IntType());
        } else {
            this.outputType = new IntType();
        }
    }

    public void addInput(InputType in) {
        inputTypes.add(in);
    }

    @Override
    public AbstractType start() {
        return new InitType(this.outputType);
    }

    @Override
    public String getName() {
        return "DeepCoderGrammar";
    }

    public AbstractType getOutputType() {
        return this.outputType;
    }

    @Override
    public List<Production<AbstractType>> getLineProductions(int size) { return null; }

    @Override
    public List<Production<AbstractType>> getInputProductions() {
        List<Production<AbstractType>> productions = new ArrayList<>();

        for (InputType input : inputTypes) {
            if (input.getType() instanceof IntType)
                productions.add(new Production<>(new IntType(), "input" + input.getIndex()));
            else if (input.getType() instanceof ListType)
                productions.add(new Production<>(new ListType(new IntType()), "input" + input.getIndex()));
            else if (input.getType() instanceof BoolType)
                productions.add(new Production<>(new BoolType(), "input" + input.getIndex()));
            else
                assert (false);
        }

        return productions;
    }

    @Override
    public List<Production<AbstractType>> getProductions() {
        List<Production<AbstractType>> productions = new ArrayList<>();

        // BoolType
        productions.add(new Production<>(new BoolType(), "true"));
        productions.add(new Production<>(new BoolType(), "false"));

        // IntType
        productions.add(new Production<>(new IntType(), "0"));
        productions.add(new Production<>(new IntType(), "1"));
        productions.add(new Production<>(new IntType(), "2"));
        productions.add(new Production<>(new IntType(), "maximum", new ListType(new IntType())));
        productions.add(new Production<>(new IntType(), "minimum", new ListType(new IntType())));
        productions.add(new Production<>(new IntType(), "sum", new ListType(new IntType())));
        productions.add(new Production<>(new IntType(), "head", new ListType(new IntType())));
        productions.add(new Production<>(new IntType(), "last", new ListType(new IntType())));
        productions.add(new Production<>(new IntType(), "count", new FunctionType(new IntType(), new BoolType()),
                new ListType(new IntType())));

        // ListType -- only considering lists of IntType
        productions.add(new Production<>(new ListType(new IntType()), "filter", new FunctionType(new IntType(), new BoolType()),
                new ListType(new IntType())));
        productions.add(new Production<>(new ListType(new IntType()), "map", new FunctionType(new IntType(), new IntType()),
                new ListType(new IntType())));
        productions.add(new Production<>(new ListType(new IntType()), "zipWith", new FunctionType(new PairType(new IntType(), new IntType()), new IntType()),
                new ListType(new IntType()), new ListType(new IntType())));
        productions.add(new Production<>(new ListType(new IntType()), "sort", new ListType(new IntType())));
        productions.add(new Production<>(new ListType(new IntType()), "reverse", new ListType(new IntType())));
        productions.add(new Production<>(new ListType(new IntType()), "take", new ListType(new IntType()), new IntType()));


        AbstractType I = new IntType();
        AbstractType O = new IntType();
        AbstractType O2 = new ListType(new IntType());
//        productions.add(new Production<>(O, "foldRight", new FunctionType(new PairType(I, O), O), O, new ListType(I)));
        productions.add(new Production<>(O2, "foldRight", new FunctionType(new PairType(I, O2), O2), O2, new ListType(I)));
        productions.add(new Production<>(new FunctionType(new PairType(I, O2), O2), "l(a,x).(cons a x)"));

//        productions.add(new Production<>(new ListType(new IntType()), "foldRight", new FunctionType(new PairType(I, O), O), O));

        //FunctionType
        productions.add(new Production<>(new FunctionType(new PairType(new IntType(), new IntType()), new IntType()), "l(a,b).(+ a b)"));
        productions.add(new Production<>(new FunctionType(new PairType(new IntType(), new IntType()), new IntType()), "l(a,b).(* a b)"));
//        productions.add(new Production<>(new FunctionType(new PairType(new IntType(), new IntType()), new IntType()), "l(a,b).(% a b)"));
        productions.add(new Production<>(new FunctionType(new PairType(new IntType(), new IntType()), new IntType()), "l(a,b).(min a b)"));

        productions.add(new Production<>(new FunctionType(new PairType(new IntType(), new IntType()), new BoolType()), "l(a,b).(> a b)"));
        productions.add(new Production<>(new FunctionType(new PairType(new IntType(), new IntType()), new BoolType()), "l(a,b).(< a b)"));
        productions.add(new Production<>(new FunctionType(new PairType(new IntType(), new IntType()), new BoolType()), "l(a,b).(== a b)"));

        productions.add(new Production<>(new FunctionType(new PairType(new BoolType(), new BoolType()), new BoolType()), "l(a,b).(|| a b)"));
        productions.add(new Production<>(new FunctionType(new PairType(new BoolType(), new BoolType()), new BoolType()), "l(a,b).(&& a b)"));

        productions.add(new Production<>(new FunctionType(new IntType(), new IntType()), "l(a).(+ a b)", new IntType()));

        productions.add(new Production<>(new FunctionType(new IntType(), new BoolType()), "l(a).(> a b)", new IntType()));
        productions.add(new Production<>(new FunctionType(new IntType(), new BoolType()), "l(a).(< a b)", new IntType()));
        productions.add(new Production<>(new FunctionType(new IntType(), new BoolType()), "l(a).(== a b)", new IntType()));
        productions.add(new Production<>(new FunctionType(new IntType(), new BoolType()), "l(a).(%!=2 a b)", new IntType()));

        return productions;
    }

    @Override
    public List<Production<AbstractType>> productionsFor(AbstractType symbol) {
        List<Production<AbstractType>> productions = new ArrayList<>();
        if (symbol instanceof InitType) {
            InitType type = (InitType) symbol;
            productions.add(new Production<>(symbol, "root", type.goalType));
        } else if (symbol instanceof BoolType) {
            productions.add(new Production<>(symbol, "true"));
            productions.add(new Production<>(symbol, "false"));
        } else if (symbol instanceof IntType) {
            productions.add(new Production<>(symbol, "0"));
            productions.add(new Production<>(symbol, "1"));
            productions.add(new Production<>(symbol, "maximum", new ListType(new IntType())));
            productions.add(new Production<>(symbol, "minimum", new ListType(new IntType())));
            productions.add(new Production<>(symbol, "sum", new ListType(new IntType())));
            productions.add(new Production<>(symbol, "head", new ListType(new IntType())));
            productions.add(new Production<>(symbol, "last", new ListType(new IntType())));

            productions.add(new Production<>(symbol, "count", new FunctionType(new IntType(), new BoolType()),
                    new ListType(new IntType())));

        } else if (symbol instanceof ListType) {
            ListType type = (ListType) symbol;
            AbstractType T = type.type;
            /* list can go to list */
            // filter (T -> Boolean) ::= (List<T> -> List<T>)
            productions.add(new Production<>(symbol, "filter", new FunctionType(T, new BoolType()),
                    new ListType(new IntType())));
            // map (I -> O) ::= (List<I> -> List<O>)
            productions.add(new Production<>(symbol, "map", new FunctionType(T, T),
                    new ListType(new IntType())));
            /* list can go to two list */
            // zipWith (List<T> -> List<T> -> int -> int -> int -> List<T>)
            productions.add(new Production<>(symbol, "zipWith", new FunctionType(new PairType(T, T), T),
                    new ListType(new IntType()), new ListType(new IntType())));

            /* other functions */
            productions.add(new Production<>(symbol, "sort", new ListType(new IntType())));
            productions.add(new Production<>(symbol, "reverse", new ListType(new IntType())));
            productions.add(new Production<>(symbol, "take", new ListType(new IntType()), new IntType()));
        } else if (symbol instanceof FunctionType) {
            FunctionType type = (FunctionType) symbol;
            // l(a,b).(+ a b) ::= ((Integer, Integer) -> Integer)
            // l(a,b).(* a b) ::= ((Integer, Integer) -> Integer)
            if (type.inputType instanceof PairType && ((PairType) type.inputType).firstType instanceof IntType
                    && ((PairType) type.inputType).secondType instanceof IntType && type.outputType instanceof IntType) {
                productions.add(new Production<>(symbol, "l(a,b).(+ a b)"));
                productions.add(new Production<>(symbol, "l(a,b).(* a b)"));
//                productions.add(new Production<>(symbol, "l(a,b).(% a b)"));
                productions.add(new Production<>(symbol, "l(a,b).(min a b)"));
            }
            // l(a,b).(> a b) ::= ((Integer, Integer) -> Boolean)
            // l(a,b).(< a b) ::= ((Integer, Integer) -> Boolean)
            // l(a,b).(>= a b) ::= ((Integer, Integer) -> Boolean)
            // l(a,b).(<= a b) ::= ((Integer, Integer) -> Boolean)
            if (type.inputType instanceof PairType && ((PairType) type.inputType).firstType instanceof IntType
                    && ((PairType) type.inputType).secondType instanceof IntType && type.outputType instanceof BoolType) {
                productions.add(new Production<>(symbol, "l(a,b).(> a b)"));
                productions.add(new Production<>(symbol, "l(a,b).(< a b)"));
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
//                productions.add(new Production<>(symbol, "l(a).(+ a b)", new IntType()));
//                productions.add(new Production<>(symbol, "l(a).(* a b)", new IntType()));
                productions.add(new Production<>(symbol, "inc3"));
//                productions.add(new Production<>(symbol, "l(a).(% a b)", new IntType()));
            }
            // l(a).(> a b) (Integer) ::= (Integer -> Boolean)
            // l(a).(< a b) (Integer) ::= (Integer -> Boolean)
            if (type.inputType instanceof IntType && type.outputType instanceof BoolType) {
                productions.add(new Production<>(symbol, "l(a).(> a b)", new IntType()));
                productions.add(new Production<>(symbol, "l(a).(< a b)", new IntType()));
                productions.add(new Production<>(symbol, "l(a).(== a b)", new IntType()));
            }
        }

        /* Handle inputs */
        for (InputType input : inputTypes) {
            boolean flag1 = (symbol instanceof IntType) && (input.getType() instanceof IntType);
            boolean flag2 = (symbol instanceof ListType) && (input.getType() instanceof ListType);
            if (flag1 || flag2) {
                productions.add(new Production<>(symbol, "input" + input.getIndex()));
            }
        }
        return productions;
    }
}
