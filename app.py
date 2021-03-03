from memory_profiler import profile


import gc
from streamlit import title, pyplot, image, button, header, file_uploader
from PIL import Image , ImageOps
from tensorflow.keras.models import load_model
from numpy import ndarray, asarray, argmax, float32
import pickle
from  matplotlib.pyplot import xlabel,yticks,ylabel,xticks, subplots
from matplotlib.pyplot import cm as color_map


label = ['Brontosaurus', 'Godzilla', 'Stegosaurus', 'T-Rex', 'Triceratops']
label_index ={'Brontosaurus':0,
             'Godzilla':1,
              'Stegosaurus':2, 
              'T-Rex':3, 
              'Triceratops':4}




def get_class(image):
    model_load = load_model('dino_classifier')
    image = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)
    data = ndarray(shape=(1, 224, 224, 3), dtype=float32)
    data[0] = asarray(image)/255
    
    predict = model_load.predict( data  )
    del model_load 
    gc.collect()
    predict = argmax(predict)
    
    return predict



def confusion_matrix(predict, actual):
    f = open('cm.pickle', 'rb')
    cm = pickle.load(f)
    f.close()
    cm[actual, predict] +=1
    f = open('cm.pickle', 'wb')
    pickle.dump(cm, f)
    f.close()
    
    fig, ax = subplots()
    ax.matshow(cm, cmap=color_map.Blues)
    yticks([0,1,2,3,4],label)
    ylabel('Actual')
    xticks([0,1,2,3,4],label, rotation=45)
    xlabel('Predict')
                
    for i in range(5):
        for j in range(5):
            ax.text(j-0.1,i,cm[i,j], c='Red')
    return fig
    


def main():
    title("Dino-Classifier")
    header("It is a T-rex, a Brontosaurus, a Stregosaurus, a Triceratops or Godzilla?  ")
    
    
    uploaded_file = file_uploader("Choose a dinosaur image, or Godzilla!", type=["png","jpeg","jpg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        predict = get_class(img)
        title(f'I think it is a {label[predict]}:')
        image(img, use_column_width=True)
        fig = None
    
        #select = selectbox( '#Choose the correct solution' ,label)
        #if select is not None:
        #    fig =confusion_matrix(predict, label_index[select])
        #    pyplot(fig)
        if fig is None:
       
            title("What is it?")
            if button('Brontosaurus'):
                fig =confusion_matrix(predict, 0)
                
            if button('Stregosaurus'):
                fig = confusion_matrix(predict, 2)
                
            if button('Triceratops'):
                fig = confusion_matrix(predict, 4)
                
            if button('T-Rex'):
                fig = confusion_matrix(predict, 3)
                
            if button('Godzilla'):
                fig = confusion_matrix(predict, 1)
        if fig is not None:
            '# Confusion Matrix'
            pyplot(fig)

main()