import React from 'react';
import {StyleShseet, Text, View, TouchableOpacity, ImageBackground, Dimensions, Button, StyleSheet} from 'react-native';
import { Camera } from 'expo-camera';
import { NavigationEvents } from "react-navigation";
import { Ionicons } from '@expo/vector-icons';
import Colors from "../constants/Colors";

export default class CameraComponent extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            hasPermission: false,
            cameraType: Camera.Constants.Type.back,
            openCamera: true,
        };
        this.toggleCamera = this.toggleCamera.bind(this);
        this.retakeImage = this.retakeImage.bind(this);
        this.sendImage = this.sendImage.bind(this);
    }

    async componentDidMount() {
        const status = await Camera.requestPermissionsAsync();
        this.setState(state => ({
            hasPermission: status.granted
        }));
    }

    async takeImage() {
        if (this.camera && this.state.hasPermission) {
            let image = await this.camera.takePictureAsync();
            this.setState(state => ({
                image: image
            }));
            this.toggleCamera();
        }
    }

    toggleCamera() {
        this.setState(state => ({
            openCamera: true
        }))
    }

    retakeImage() {
        this.setState(state => ({
            image: null,
            openCamera: true,
        }))
    }

    sendImage() {
        console.log("send");
        this.setState(state => ({
            image: null,
            openCamera: false,
        }));
        this.props.navigation.navigate("Prices", {imageTaken: true});
    }

    render() {
        if (this.state.image) {
            return(
                <View style={{ flex: 1 }}>
                    <ImageBackground
                        style={{
                            flex: 1,
                            justifyContent: "flex-end",
                            width: Dimensions.get('window').width,
                            height: Dimensions.get('window').height
                        }}
                        source={{uri: this.state.image.uri}}
                    >
                        <View style={{flex: 0.1, justifyContent: "space-around", flexDirection: "row"}}>
                            {/*<Button onPress={this.retakeImage} title={"Retake image"} />*/}
                            {/*<Button onPress={this.retakeImage} title={"Send image"} />*/}
                            <Ionicons
                                name={"md-redo"}
                                size={50}
                                // style={{ marginBottom: -3 }}
                                color={"white"}
                                onPress={this.retakeImage}
                            />
                            <Ionicons
                                name={"md-send"}
                                size={50}
                                // style={{ marginBottom: -3 }}
                                color={"white"}
                                onPress={this.sendImage}
                            />
                        </View>
                    </ImageBackground>
                    {/*<Button onPress={this.} title={"Camera"} />*/}
                </View>
            )
        } else if(this.state.openCamera) {
            return(
                <View style={{ flex: 1 }}>
                    <NavigationEvents
                        onDidBlur={payload => this.setState(state => ({
                            openCamera: false,
                        }))}>
                    </NavigationEvents>
                    <Camera
                        style={{ flex: 1, justifyContent: "flex-end" }}
                        type={this.state.cameraType}
                        ref={ref => {this.camera = ref}}
                    >
                        <TouchableOpacity
                            style={{
                                flex: 0.1,
                                // borderWidth: 3,
                                // borderColor: "black",
                                alignItems: 'center',
                                justifyContent: "center"
                            }}
                            onPress={() => {
                                this.takeImage();
                            }}>
                            <Ionicons
                                name={"md-radio-button-off"}
                                size={50}
                                // style={{ marginBottom: -3 }}
                                color={"white"}
                            />
                            {/*<Text style={{ fontSize: 18, marginBottom: 10, color: 'red' }}> Cliick </Text>*/}
                        </TouchableOpacity>
                    </Camera>
                </View>
            )
        } else {
            return(
                <NavigationEvents
                    onWillFocus={payload => this.setState(state => ({
                        openCamera: true,
                    }))}>
                </NavigationEvents>
            )
        }
    }
}

const styles = StyleSheet.create({
    imageButton: {
        flex: 1
    },
});